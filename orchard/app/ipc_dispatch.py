from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import weakref
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pynng

from orchard import defaults
from orchard.engine.global_context import GlobalContext
from orchard.engine.multiprocess import pid_is_alive, read_pid_file
from orchard.ipc.utils import ResponseDeltaDict

logger = logging.getLogger(__name__)

EVENT_TOPIC_PREFIX = b"__PIE_EVENT__:"
ENGINE_LIVENESS_POLL_INTERVAL_S = 5.0
RESPONSE_RECV_TIMEOUT_MS = 1000


class IPCDispatcher:
    """
    Simple prefix-based dispatcher for IPC(Inter-Process Communication) messages.
    """

    def __init__(self) -> None:
        self._handlers: list[tuple[bytes, Callable[[IPCState, bytes], None]]] = []

    def register_handler(
        self,
        prefix: bytes,
        handler: Callable[[IPCState, bytes], None],
    ) -> None:
        self._handlers.append((prefix, handler))

    def dispatch(self, ipc_state: IPCState, msg_bytes: bytes) -> bool:
        for prefix, handler in self._handlers:
            if msg_bytes.startswith(prefix):
                handler(ipc_state, msg_bytes)
                return True
        return False


@dataclass(slots=True)
class QueueRegistration:
    """Holds the necessary context to safely dispatch a delta to a client."""

    loop: asyncio.AbstractEventLoop
    queue: asyncio.Queue[ResponseDeltaDict]


class IPCState:
    """
    Holds the process-wide state for IPC components, including NNG sockets
    and active request queues.
    """

    def __init__(self, global_context: GlobalContext):
        # NNG sockets, initialized by InferenceEngine
        self.request_socket: pynng.Push0 | None = None
        self.response_socket: pynng.Sub0 | None = None
        self.management_socket: pynng.Req0 | None = None
        self._management_lock = asyncio.Lock()
        self._management_lock_loop: asyncio.AbstractEventLoop | None = None

        self.response_channel_id: int = 0
        self.active_request_queues: dict[int, QueueRegistration] = {}

        self.request_id_counter: int = 0
        self.dispatcher_task: asyncio.Task | None = None
        self._request_id_lock = threading.Lock()
        self.response_topic_prefix: bytes = b""
        self.response_topic_prefix_len: int = 0
        self.engine_pid_file: Path | None = None
        self.engine_dead: bool = False
        self.shutdown_requested: bool = False

        # Monotonic time of the last engine message that proves generation or
        # activation progress (any response delta, model_loaded/_load_failed).
        # Written by the dispatcher thread, read by delta waiters; a float
        # attribute write is atomic under the GIL. Telemetry heartbeats do not
        # count — a wedged engine still emits them.
        self.last_progress_monotonic: float = 0.0

        self._inflight_lock = threading.Lock()
        self._inflight_ops = 0
        self._inflight_drained = threading.Event()
        self._inflight_drained.set()

        self.global_context = weakref.ref(global_context)

    @property
    def management_lock(self) -> asyncio.Lock:
        # An asyncio.Lock binds to the loop that first acquires it, but this
        # state outlives test/session event loops (model activation from a
        # later loop raised "bound to a different event loop"). Rebind when
        # the loop changed and the lock is free; real cross-loop contention
        # stays a loud error.
        loop = asyncio.get_running_loop()
        if self._management_lock_loop is not loop:
            if self._management_lock.locked():
                raise RuntimeError("management_lock is held on another event loop")
            self._management_lock = asyncio.Lock()
            self._management_lock_loop = loop
        return self._management_lock

    @contextmanager
    def socket_op(self):
        """Tracks an in-flight NNG operation so shutdown can drain before
        closing the sockets. Refuses new operations once shutdown started;
        closing a socket while an async operation is pending frees nng aio
        structures still in use and aborts the process."""
        with self._inflight_lock:
            if self.shutdown_requested:
                raise RuntimeError(
                    "IPC shutdown in progress; cannot start new socket operations."
                )
            self._inflight_ops += 1
            self._inflight_drained.clear()
        try:
            yield
        finally:
            with self._inflight_lock:
                self._inflight_ops -= 1
                if self._inflight_ops == 0:
                    self._inflight_drained.set()

    def wait_for_inflight_drain(self, timeout_s: float) -> bool:
        """Blocks until no socket operations are in flight. Returns False on
        timeout, meaning the NNG sockets are not safe to close."""
        return self._inflight_drained.wait(timeout_s)

    async def send_request(self, request_bytes: bytes) -> None:
        socket = self.request_socket
        if socket is None:
            raise RuntimeError("Request socket is not initialized.")
        if self.engine_dead:
            raise RuntimeError("Engine process is dead; cannot submit new requests.")
        with self.socket_op():
            await socket.asend(request_bytes)

    async def get_next_request_id(self) -> int:
        """Atomically increments and returns the next request ID."""
        with self._request_id_lock:
            self.request_id_counter += 1
            # Basic overflow protection
            if self.request_id_counter >= 2**63:
                self.request_id_counter = 1
            return self.request_id_counter

    async def next_delta(
        self,
        queue: asyncio.Queue[ResponseDeltaDict],
    ) -> ResponseDeltaDict:
        """Waits for this request's next delta without mistaking a busy engine
        for a dead one.

        A cold-peak volley legitimately parks a request's first prefill far
        beyond any flat per-delta timeout (measured 91s behind ~800 queued
        sequences at buckshot width 21) while the engine streams deltas for
        other requests the whole time. So wait in slices: engine-wide
        delta/activation traffic keeps the wait alive, DELTA_TIMEOUT_S of
        engine-wide silence raises TimeoutError (the old flat-timeout
        semantics for a wedged engine), and DELTA_HARD_TIMEOUT_S bounds the
        total wait (a request the engine dropped). Engine death needs no
        timeout at all: the dispatcher's exit path fails every registered
        queue with a terminal error delta immediately.
        """
        start = time.monotonic()
        hard_deadline = start + defaults.DELTA_HARD_TIMEOUT_S
        while True:
            if self.engine_dead and queue.empty():
                raise TimeoutError(
                    "Engine process died while waiting for a response delta."
                )
            now = time.monotonic()
            newest_progress = max(start, self.last_progress_monotonic)
            deadline = min(newest_progress + defaults.DELTA_TIMEOUT_S, hard_deadline)
            if now >= deadline:
                raise TimeoutError(
                    "Timed out waiting for a response delta from the engine."
                )
            try:
                return await asyncio.wait_for(queue.get(), timeout=deadline - now)
            except TimeoutError:
                continue

    async def send_management_command(
        self,
        command: dict[str, Any],
        *,
        timeout: float | None = 2.0,
    ) -> dict[str, Any]:
        socket = self.management_socket
        if socket is None:
            raise RuntimeError("Management socket is not initialized.")
        if self.engine_dead:
            raise RuntimeError(
                "Engine process is dead; cannot send management commands."
            )

        payload = json.dumps(command).encode("utf-8")
        with self.socket_op():
            async with self.management_lock:
                await socket.asend(payload)
                if timeout is None:
                    reply_bytes = await socket.arecv()
                else:
                    reply_bytes = await asyncio.wait_for(socket.arecv(), timeout)

        try:
            response = json.loads(reply_bytes.decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(
                "Engine returned malformed management response."
            ) from exc

        if not isinstance(response, dict):
            raise RuntimeError("Engine returned malformed management response.")
        return response

    async def cancel_request(
        self,
        request_id: int,
        *,
        timeout: float | None = 2.0,
    ) -> dict[str, Any]:
        response = await self.send_management_command(
            {"type": "cancel_request", "request_id": request_id},
            timeout=timeout,
        )
        status = str(response.get("status") or "").lower()
        if status not in {"ok", "accepted"}:
            message = response.get("message", "unknown error")
            raise RuntimeError(
                f"Engine rejected cancel_request for '{request_id}': {message}"
            )
        return response

    async def cancel_model_load(
        self,
        requested_id: str,
        *,
        canonical_id: str | None = None,
        timeout: float | None = 2.0,
    ) -> dict[str, Any]:
        command = {
            "type": "cancel_model_load",
            "requested_id": requested_id,
        }
        if canonical_id:
            command["canonical_id"] = canonical_id

        response = await self.send_management_command(command, timeout=timeout)
        status = str(response.get("status") or "").lower()
        if status not in {"ok", "accepted"}:
            message = response.get("message", "unknown error")
            raise RuntimeError(
                f"Engine rejected cancel_model_load for '{requested_id}': {message}"
            )
        return response

    def handle_response_delta(self, msg_bytes: bytes) -> None:
        """Dispatches a response delta to the registered client queue."""
        self.last_progress_monotonic = time.monotonic()
        prefix_len = self.response_topic_prefix_len
        if prefix_len <= 0:
            logger.error(
                "Response topic prefix uninitialized; dropping response delta."
            )
            return

        json_body = msg_bytes[prefix_len:]
        try:
            payload: ResponseDeltaDict = json.loads(json_body)
        except json.JSONDecodeError as exc:
            snippet = json_body[:256].decode("utf-8", errors="replace")
            logger.error(
                "Failed to parse response delta JSON: %s | payload snippet: %s",
                exc,
                snippet,
            )
            return

        request_id = payload.get("request_id")

        if request_id is None:
            logger.warning("Received response delta with no request_id.")
            return

        if registration := self.active_request_queues.get(request_id):
            registration.loop.call_soon_threadsafe(
                registration.queue.put_nowait,
                payload,
            )
        else:
            logger.debug(
                "Received delta for unknown/completed request_id %d. Discarding.",
                request_id,
            )

    def handle_engine_event(self, msg_bytes: bytes) -> None:
        """Handles engine events broadcasted from the engine."""
        parts = msg_bytes.split(b"\x00", 1)
        if len(parts) != 2:
            utf8_body = msg_bytes.decode("utf-8", errors="replace")
            logger.warning("Received malformed event message: %s", utf8_body)
            return

        topic_part, json_body = parts
        event_name = topic_part[len(EVENT_TOPIC_PREFIX) :].decode("utf-8")
        ctx = self.global_context()
        try:
            payload = json.loads(json_body)
        except Exception as e:
            logger.error("Failed to parse engine event payload: %s", e)
            return

        if event_name == "telemetry" and ctx is not None:
            ctx.last_telemetry = payload
            return

        if event_name in ("model_loaded", "model_load_failed"):
            self.last_progress_monotonic = time.monotonic()

        if event_name == "model_loaded":
            model_id = payload.get("model_id")
            if not model_id:
                logger.warning("Received model_loaded event without model_id.")
                return

            if ctx and ctx.model_registry:
                try:
                    ctx.model_registry.handle_model_loaded(payload)
                except Exception:
                    logger.exception(
                        "Failed to handle model_loaded event for '%s'.", model_id
                    )
            else:
                logger.warning(
                    "Received model_loaded but no model registry is available."
                )
            return

        if event_name == "model_load_failed":
            model_id = payload.get("model_id")
            if not model_id:
                logger.warning("Received model_load_failed event without model_id.")
                return

            if ctx and ctx.model_registry:
                try:
                    ctx.model_registry.handle_model_load_failed(payload)
                except Exception:
                    logger.exception(
                        "Failed to handle model_load_failed event for '%s'.",
                        model_id,
                    )
            else:
                logger.warning(
                    "Received model_load_failed but no model registry is available."
                )
            return

        logger.warning("Received unknown engine event '%s'.", event_name)

    def engine_process_is_alive(self) -> bool:
        if self.engine_pid_file is None:
            return True

        pid = read_pid_file(self.engine_pid_file)
        return pid is not None and pid_is_alive(pid)

    @staticmethod
    async def run_ipc_listener(ipc_state: IPCState) -> None:
        """
        Asynchronously consumes messages from the NNG SUB socket and dispatches
        them to the appropriate client queues or event waiters.
        """
        logger.info("NNG response dispatcher task starting...")

        sub_socket = ipc_state.response_socket
        if not sub_socket:
            logger.critical("Response socket not initialized. Dispatcher cannot run.")
            return
        sub_socket.recv_timeout = RESPONSE_RECV_TIMEOUT_MS

        dispatcher = IPCDispatcher()
        resp_topic_prefix = f"resp:{ipc_state.response_channel_id:x}:".encode()
        ipc_state.response_topic_prefix = resp_topic_prefix
        ipc_state.response_topic_prefix_len = len(resp_topic_prefix)

        dispatcher.register_handler(resp_topic_prefix, IPCState.handle_response_delta)
        dispatcher.register_handler(EVENT_TOPIC_PREFIX, IPCState.handle_engine_event)

        try:
            last_engine_check = 0.0
            while True:
                if ipc_state.shutdown_requested:
                    logger.info("Dispatcher shutdown requested; exiting IPC listener.")
                    break

                now = time.monotonic()
                if now - last_engine_check >= ENGINE_LIVENESS_POLL_INTERVAL_S:
                    last_engine_check = now
                    try:
                        engine_alive = ipc_state.engine_process_is_alive()
                    except Exception:
                        # A probe that failed to run is no death verdict. Only a
                        # confirmed-dead engine may exit the dispatcher, which
                        # fails every active stream and latches engine_dead.
                        logger.exception(
                            "Engine liveness probe failed; retrying at next poll."
                        )
                        engine_alive = True
                    if not engine_alive:
                        logger.error(
                            "PIE is no longer alive; shutting down response dispatcher."
                        )
                        break
                try:
                    msg = await sub_socket.arecv_msg()
                    if not dispatcher.dispatch(ipc_state, msg.bytes):
                        logger.warning("Received IPC message with unregistered prefix.")
                except pynng.Timeout:
                    continue
                except pynng.Closed:
                    logger.info("Response socket closed, dispatcher shutting down.")
                    break
                except Exception:
                    logger.exception("Unexpected error in NNG message reception loop.")
                    break
        except asyncio.CancelledError:
            logger.info("Response dispatcher task was cancelled.")
        finally:
            if ipc_state.active_request_queues:
                logger.warning(
                    "Response dispatcher exiting with %d active request queues; failing them.",
                    len(ipc_state.active_request_queues),
                )

            # Flush active response queues with a terminal error delta so callers can complete.
            error_payload = {
                "is_final_delta": True,
                "finish_reason": "error",
                "content": "Engine process disconnected.",
                "error": "Engine process disconnected.",
            }
            for request_id, registration in list(
                ipc_state.active_request_queues.items()
            ):
                payload = {"request_id": request_id, **error_payload}
                try:
                    registration.loop.call_soon_threadsafe(
                        registration.queue.put_nowait,
                        payload,
                    )
                except Exception:
                    logger.exception(
                        "Failed to enqueue terminal error delta for request %d.",
                        request_id,
                    )
            ipc_state.active_request_queues.clear()
            ipc_state.engine_dead = True
            logger.info("Response dispatcher task finished.")
