import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING

from huggingface_hub import snapshot_download

from orchard.app.model_resolver import (
    ModelResolutionError,
    ModelResolver,
    ResolvedModel,
)
from orchard.engine.multiprocess import pid_is_alive, read_pid_file
from orchard.formatter import ChatFormatter

if TYPE_CHECKING:
    from orchard.app.ipc_dispatch import IPCState

logger = logging.getLogger(__name__)


class ModelLoadState(Enum):
    IDLE = auto()
    DOWNLOADING = auto()
    ACTIVATING = auto()
    LOADING = auto()
    READY = auto()
    FAILED = auto()


@dataclass(slots=True)
class ModelInfo:
    model_id: str
    model_path: str
    formatter: ChatFormatter
    capabilities: dict[str, list[int]] | None = None


@dataclass(slots=True)
class ModelEntry:
    state: ModelLoadState = ModelLoadState.IDLE
    info: ModelInfo | None = None
    error: str | None = None
    event: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task[None] | None = None
    resolved: ResolvedModel | None = None
    bytes_downloaded: int | None = None
    bytes_total: int | None = None
    activation_future: asyncio.Future | None = None
    activation_loop: asyncio.AbstractEventLoop | None = None


class ModelRegistry:
    def __init__(self, ipc_state: "IPCState") -> None:
        self._entries: dict[str, ModelEntry] = {}
        self._lock = asyncio.Lock()
        self._resolver = ModelResolver()
        self._alias_cache: dict[str, str] = {}
        self._ipc_state = ipc_state

    async def ensure_ready(
        self, requested_model_id: str, *, timeout: float | None = None
    ) -> ModelInfo:
        """Ensure a model is loaded and return the ModelInfo."""
        return await self.ensure_loaded(requested_model_id, timeout=timeout)

    def ensure_ready_sync(
        self, requested_model_id: str, *, timeout: float | None = None
    ) -> ModelInfo:
        """Blocking wrapper around ensure_ready for synchronous contexts."""

        return asyncio.run(self.ensure_ready(requested_model_id, timeout=timeout))

    async def ensure_loaded(
        self, requested_model_id: str, timeout: float | None = None
    ) -> ModelInfo:
        """Ensure a model is staged locally and activated on the engine."""
        state, canonical_id = await self.schedule_model(requested_model_id)
        # Wait for local readiness (download + formatter)
        state, info, error = await self.await_model(canonical_id, timeout)
        if state == ModelLoadState.FAILED or not info:
            detail = error or f"Model '{canonical_id}' failed to load."
            raise RuntimeError(detail)
        if state == ModelLoadState.READY:
            return info

        # Activation phase (IPC load command)
        async with self._lock:
            entry = self._entries[canonical_id]
            if entry.state == ModelLoadState.READY and entry.info:
                return entry.info

            loop = asyncio.get_running_loop()
            should_send_command = True
            # Reuse an in-flight activation if one exists
            if entry.state == ModelLoadState.ACTIVATING and entry.activation_future:
                waiter = entry.activation_future
                should_send_command = False
            else:
                # Cancel any stale waiter
                if entry.activation_future and not entry.activation_future.done():
                    entry.activation_future.cancel()
                entry.activation_loop = loop
                entry.activation_future = loop.create_future()
                entry.state = ModelLoadState.ACTIVATING
                waiter = entry.activation_future

        if should_send_command:
            await self._send_load_model_command(
                requested_id=requested_model_id,
                canonical_id=canonical_id,
                info=info,
            )

        try:
            await self._await_activation_with_engine_liveness(
                canonical_id=canonical_id,
                waiter=waiter,
                timeout=timeout,
            )
        except Exception as exc:
            async with self._lock:
                entry = self._entries.get(canonical_id)
                if entry and entry.state != ModelLoadState.READY:
                    entry.state = ModelLoadState.FAILED
                    entry.error = str(exc)
                    entry.activation_future = None
                    entry.activation_loop = None
            raise

        ready_info = self.get_if_ready(canonical_id)
        if not ready_info:
            raise RuntimeError(f"Model '{canonical_id}' failed to activate.")
        return ready_info

    async def schedule_model(
        self, requested_model_id: str, *, force_reload: bool = False
    ) -> tuple[ModelLoadState, str]:
        """Ensure a model is loading or ready and return the canonical identifier."""

        resolved = self._resolver.resolve(requested_model_id)
        canonical_id = resolved.canonical_id
        self._alias_cache[requested_model_id.lower()] = canonical_id
        self._alias_cache.setdefault(canonical_id.lower(), canonical_id)

        async with self._lock:
            entry = self._entries.get(canonical_id)
            if entry is None:
                entry = ModelEntry()
                self._entries[canonical_id] = entry

            if entry.state == ModelLoadState.READY and not force_reload:
                return ModelLoadState.READY, canonical_id

            if (
                entry.state
                in (
                    ModelLoadState.LOADING,
                    ModelLoadState.DOWNLOADING,
                    ModelLoadState.ACTIVATING,
                )
                and not force_reload
            ):
                return entry.state, canonical_id

            if entry.state == ModelLoadState.FAILED and not force_reload:
                return ModelLoadState.FAILED, canonical_id

            entry.error = None
            entry.info = None
            entry.event = asyncio.Event()
            entry.resolved = resolved
            entry.bytes_downloaded = None
            entry.bytes_total = None
            entry.activation_future = None
            entry.activation_loop = None

            # If the model is already local (or in HF cache), build formatter immediately.
            if resolved.source in {"local", "hf_cache"} or (
                resolved.model_path and (resolved.model_path / "config.json").exists()
            ):
                try:
                    formatter = await asyncio.to_thread(
                        ChatFormatter, str(resolved.model_path)
                    )
                    info = ModelInfo(
                        model_id=resolved.canonical_id,
                        model_path=str(resolved.model_path),
                        formatter=formatter,
                    )
                    entry.info = info
                    entry.state = ModelLoadState.LOADING
                    entry.event.set()
                    return entry.state, canonical_id
                except Exception as exc:
                    entry.error = str(exc)
                    entry.state = ModelLoadState.FAILED
                    entry.event.set()
                    return ModelLoadState.FAILED, canonical_id

            # Otherwise, schedule async download + formatter build.
            entry.state = ModelLoadState.DOWNLOADING

            async def loader() -> None:
                loop = asyncio.get_running_loop()

                async def _update_progress(
                    bytes_downloaded: int, bytes_total: int
                ) -> None:
                    async with self._lock:
                        entry.bytes_downloaded = bytes_downloaded
                        entry.bytes_total = bytes_total

                def _progress_cb(bytes_downloaded: int, bytes_total: int) -> None:
                    loop.call_soon_threadsafe(
                        asyncio.create_task,
                        _update_progress(bytes_downloaded, bytes_total),
                    )

                try:
                    download_path = await asyncio.to_thread(
                        snapshot_download,
                        resolved.hf_repo or resolved.canonical_id,
                        local_files_only=False,
                        progress_callback=_progress_cb,  # type: ignore [reportCallIssue]
                    )
                    await _update_progress(
                        entry.bytes_total or 0, entry.bytes_total or 0
                    )
                    formatter = await asyncio.to_thread(
                        ChatFormatter, str(download_path)
                    )
                    info = ModelInfo(
                        model_id=resolved.canonical_id,
                        model_path=str(download_path),
                        formatter=formatter,
                    )
                    async with self._lock:
                        entry.info = info
                        entry.state = ModelLoadState.LOADING
                        entry.resolved = ResolvedModel(
                            canonical_id=resolved.canonical_id,
                            model_path=Path(download_path),
                            source="hf_cache",
                            metadata=resolved.metadata,
                            hf_repo=resolved.hf_repo,
                        )
                except (
                    asyncio.CancelledError
                ):  # pragma: no cover - cooperative cancellation
                    raise
                except Exception as exc:  # pragma: no cover - best effort logging
                    async with self._lock:
                        entry.error = str(exc)
                        entry.state = ModelLoadState.FAILED
                        entry.info = None
                finally:
                    entry.event.set()

            entry.task = asyncio.create_task(loader())
            return ModelLoadState.DOWNLOADING, canonical_id

    async def await_model(
        self, model_id: str, timeout: float | None = None
    ) -> tuple[ModelLoadState, ModelInfo | None, str | None]:
        canonical_id = self._canonicalize(model_id)
        if canonical_id is None:
            raise ValueError(f"Model '{model_id}' has not been scheduled")

        async with self._lock:
            entry = self._entries.get(canonical_id)
            if entry is None:
                raise ValueError(f"Model '{canonical_id}' has not been scheduled")
            event = entry.event

        try:
            if timeout is None:
                await event.wait()
            else:
                await asyncio.wait_for(event.wait(), timeout)
        except TimeoutError:
            return ModelLoadState.LOADING, None, None

        async with self._lock:
            entry = self._entries[canonical_id]
            return entry.state, entry.info, entry.error

    def get_if_ready(self, model_id: str) -> ModelInfo | None:
        canonical_id = self._canonicalize(model_id)
        if canonical_id is None:
            return None
        entry = self._entries.get(canonical_id)
        if entry and entry.state == ModelLoadState.READY:
            return entry.info
        return None

    def get_status(
        self, model_id: str
    ) -> tuple[ModelLoadState, str | None, dict | None]:
        canonical_id = self._canonicalize(model_id)
        if canonical_id is None:
            return ModelLoadState.IDLE, None, None
        entry = self._entries.get(canonical_id)
        if entry is None:
            return ModelLoadState.IDLE, None, None
        progress = None
        if entry.bytes_downloaded is not None or entry.bytes_total is not None:
            progress = {
                "bytes_downloaded": entry.bytes_downloaded,
                "bytes_total": entry.bytes_total,
            }
        return entry.state, entry.error, progress

    def get_error(self, model_id: str) -> str | None:
        canonical_id = self._canonicalize(model_id)
        if canonical_id is None:
            return None
        entry = self._entries.get(canonical_id)
        if entry and entry.state == ModelLoadState.FAILED:
            return entry.error
        return None

    def list_models(self) -> list[dict[str, str]]:
        """List all currently loaded models."""
        catalog = []
        for canonical_id, entry in self._entries.items():
            if entry.resolved is None:
                continue
            payload = dict(entry.resolved.metadata)
            payload["canonical_id"] = canonical_id
            payload["model_path"] = str(entry.resolved.model_path)
            payload["source"] = entry.resolved.source
            payload["hf_repo"] = entry.resolved.hf_repo or ""
            payload["state"] = entry.state.name
            catalog.append(payload)
        return catalog

    def resolve(self, model_id: str) -> ResolvedModel:
        return self._resolver.resolve(model_id)

    async def get_info(self, model_id: str) -> ModelInfo:
        if (
            model_id in self._alias_cache
            and (canonical := self._alias_cache[model_id]) in self._entries
            and (info := self._entries[canonical].info) is not None
        ):
            return info

        return await self.ensure_ready(model_id)

    async def _send_load_model_command(
        self,
        *,
        requested_id: str,
        canonical_id: str,
        info: ModelInfo,
    ) -> None:
        """Issue the IPC load_model command and handle immediate responses."""
        ipc_state = self._ipc_state
        if not ipc_state or not ipc_state.management_socket:
            raise RuntimeError("IPC state is not initialized.")

        async with self._lock:
            entry = self._entries.get(canonical_id)
        if entry is None or entry.activation_future is None:
            await self._mark_activation_failed(
                canonical_id, "Activation waiter missing."
            )
            raise RuntimeError("Activation waiter missing.")

        command = {
            "type": "load_model",
            "requested_id": requested_id,
            "canonical_id": canonical_id,
            "model_path": info.model_path,
            "wait_for_completion": False,
        }
        payload = json.dumps(command).encode("utf-8")

        management_socket = ipc_state.management_socket
        assert management_socket is not None  # for type-checkers

        try:
            async with ipc_state.management_lock:
                await management_socket.asend(payload)
                reply_bytes = await management_socket.arecv()
        except Exception as exc:
            await self._mark_activation_failed(
                canonical_id, f"Failed to send load_model command: {exc}"
            )
            raise RuntimeError(
                f"Failed to send load_model command for '{requested_id}': {exc}"
            ) from exc

        try:
            response = json.loads(reply_bytes.decode("utf-8"))
        except Exception as exc:
            await self._mark_activation_failed(
                canonical_id, "Engine returned malformed management response."
            )
            raise RuntimeError(
                "Engine returned malformed management response."
            ) from exc

        status = response.get("status")
        if status == "ok":
            capabilities = self._parse_capabilities(response)
            try:
                self.update_capabilities(canonical_id, capabilities)
            finally:
                await self._complete_activation(canonical_id, capabilities is not None)
            return

        if status != "accepted":
            message = response.get("message", "unknown error")
            await self._mark_activation_failed(
                canonical_id, f"Engine rejected load_model: {message}"
            )
            raise RuntimeError(
                f"Engine rejected load_model for '{requested_id}': {message}"
            )

        # Accepted: wait for model_loaded event via handle_model_loaded
        async with self._lock:
            entry = self._entries[canonical_id]
            waiter = entry.activation_future

        if waiter is None:
            await self._mark_activation_failed(
                canonical_id, "Activation waiter missing."
            )
            raise RuntimeError("Activation waiter missing.")

    def update_capabilities(self, model_id: str, capabilities: dict | None) -> None:
        """Update persisted capabilities for a given model."""
        if not capabilities:
            return

        canonical_id = self._canonicalize(model_id) or model_id
        entry = self._entries.get(canonical_id)
        if not entry or entry.info is None:
            logger.warning("Received capabilities for unknown model_id '%s'.", model_id)
            return

        normalized: dict[str, list[int]] = {}
        for name, value in capabilities.items():
            if isinstance(value, list | tuple):
                normalized[str(name)] = [int(v) for v in value]
            else:
                normalized[str(name)] = [int(value)]

        entry.info.capabilities = normalized

    def handle_model_loaded(self, payload: dict) -> None:
        """Handle model_loaded events emitted by the engine."""
        model_id = payload.get("model_id")
        if not model_id:
            logger.warning("Received model_loaded event without model_id.")
            return

        capabilities = payload.get("capabilities")
        try:
            self.update_capabilities(model_id, capabilities)
        except Exception:
            logger.exception("Failed to update capabilities for model '%s'.", model_id)

        entry = self._entries.get(model_id)
        if not entry:
            logger.debug("Received model_loaded for unknown model '%s'.", model_id)
            return

        waiter = entry.activation_future
        loop = entry.activation_loop
        if entry.state != ModelLoadState.ACTIVATING:
            return

        entry.state = ModelLoadState.READY
        if loop and waiter and not waiter.done():
            loop.call_soon_threadsafe(waiter.set_result, payload)
        entry.activation_future = None
        entry.activation_loop = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _engine_pid_file() -> Path:
        home = Path.home()
        mac_cache = home / "Library" / "Caches"
        base = mac_cache if mac_cache.exists() else home / ".cache"
        return base / "com.theproxycompany" / "engine.pid"

    @classmethod
    def _engine_pid_alive(cls) -> tuple[bool | None, int | None]:
        pid_file = cls._engine_pid_file()
        pid = read_pid_file(pid_file)
        if pid is None:
            return None, None
        return pid_is_alive(pid), pid

    async def _await_activation_with_engine_liveness(
        self,
        *,
        canonical_id: str,
        waiter: asyncio.Future,
        timeout: float | None,
    ) -> None:
        poll_interval_s = 0.25
        deadline = None if timeout is None else (time.monotonic() + timeout)

        while True:
            if waiter.done():
                await waiter
                return

            alive, pid = self._engine_pid_alive()
            if alive is False:
                pid_suffix = f" (pid={pid})" if pid else ""
                raise RuntimeError(
                    f"Engine process exited while activating '{canonical_id}'{pid_suffix}."
                )

            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Timed out waiting for model '{canonical_id}' activation."
                    )
                step_timeout = min(poll_interval_s, remaining)
            else:
                step_timeout = poll_interval_s

            try:
                await asyncio.wait_for(asyncio.shield(waiter), timeout=step_timeout)
                await waiter
                return
            except TimeoutError:
                continue

    def _parse_capabilities(self, response: dict) -> dict | None:
        try:
            data_field = response.get("data", {})
            if isinstance(data_field, dict):
                load_model_data = data_field.get("load_model") or {}
                if isinstance(load_model_data, dict):
                    return load_model_data.get("capabilities")
        except Exception:
            logger.exception("Failed to parse capabilities from management response.")
        return None

    async def _complete_activation(
        self, model_id: str, capabilities_updated: bool
    ) -> None:
        async with self._lock:
            entry = self._entries.get(model_id)
            if not entry:
                return
            entry.state = ModelLoadState.READY
            if (
                entry.activation_future
                and not entry.activation_future.done()
                and entry.activation_loop
            ):
                entry.activation_loop.call_soon_threadsafe(
                    entry.activation_future.set_result,
                    {"capabilities": capabilities_updated},
                )
            entry.activation_future = None
            entry.activation_loop = None

    async def _mark_activation_failed(self, model_id: str, reason: str) -> None:
        async with self._lock:
            entry = self._entries.get(model_id)
            if not entry:
                return
            entry.error = reason
            entry.state = ModelLoadState.FAILED
            if entry.activation_future and not entry.activation_future.done():
                if entry.activation_loop:
                    entry.activation_loop.call_soon_threadsafe(
                        entry.activation_future.set_exception, RuntimeError(reason)
                    )
                else:
                    entry.activation_future.set_exception(RuntimeError(reason))
            entry.activation_future = None
            entry.activation_loop = None

    def _canonicalize(self, model_id: str) -> str | None:
        """Convert a model ID to its canonical form if known."""
        if model_id in self._entries:
            return model_id
        lower = model_id.lower()
        if lower in self._alias_cache:
            return self._alias_cache[lower]
        return None


__all__ = [
    "ModelLoadState",
    "ModelRegistry",
    "ModelResolutionError",
]
