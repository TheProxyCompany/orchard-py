"""Full-duplex audio through PIE's native Moshi/Mimi architecture.

This module owns a Rust Orchard transport process. PIE owns model tensors,
inference, and residency. Output text belongs to the assistant's speech;
it is never a microphone transcript.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import shutil
from collections import deque
from collections.abc import AsyncIterator, Iterable
from importlib.resources import files
from pathlib import Path
from typing import Any, Self

import yaml

SAMPLE_RATE = 24000
FRAME_SAMPLES = 1920
DEFAULT_MODEL = "kyutai/moshika-rag-candle-bf16"
_EPOCH_EVENTS = frozenset(
    {
        "audio",
        "text_delta",
        "speech_queued",
        "speech_done",
        "interrupted",
        "reset",
        "reference_queued",
        "reference_applied",
        "retrieval_requested",
    }
)


def capabilities() -> dict[str, Any]:
    """Return the shared Pantheon catalog; native PIE currently supports MoshiRAG."""
    path = files("orchard").joinpath("formatter/profiles/moshi/capabilities.yaml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _binary(
    explicit: str | os.PathLike[str] | None, program: str, environment: str
) -> str:
    candidate = explicit or os.environ.get(environment)
    if candidate:
        path = Path(candidate).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
        raise FileNotFoundError(f"Orchard duplex executable is unavailable: {path}")
    local_build = os.environ.get("PIE_LOCAL_BUILD")
    if local_build:
        for path in (Path(local_build) / "bin" / program, Path(local_build) / program):
            if path.is_file() and os.access(path, os.X_OK):
                return str(path)
    found = shutil.which(program)
    if found:
        return found
    raise FileNotFoundError(
        f"The native Orchard audio executable is not installed. Set {environment} "
        f"to the bundled {program}, or build the matching orchard-rs native binary."
    )


class _NativeAudioSession:
    """One independently streamed microphone/output session with bounded queues.

    Use ``await DuplexSession.open(...)`` or ``await client.audio.duplex(...)``.
    Sending audio never waits for model inference. A control acknowledgement
    gives the current epoch, letting playback discard obsolete audio immediately.
    """

    _program = "orchard-duplex"
    _environment = "ORCHARD_DUPLEX_BINARY"
    _default_model = DEFAULT_MODEL

    @classmethod
    def effective_options(cls, options: dict[str, Any] | None) -> dict[str, Any]:
        return dict(options or {})

    @classmethod
    async def prepare(
        cls,
        model_id: str | None = None,
        *,
        binary: str | os.PathLike[str] | None = None,
        options: dict[str, Any] | None = None,
        ready_timeout: float = 300.0,
    ) -> dict[str, Any]:
        """Wait for PIE residency using the session's adapter, without opening it."""
        model_id = model_id or cls._default_model
        command = [
            _binary(binary, cls._program, cls._environment),
            "--model",
            model_id,
            "--prepare-only",
            "--options-json",
            json.dumps(cls.effective_options(options), allow_nan=False),
        ]
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=1_048_576,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), ready_timeout
            )
            if process.returncode != 0:
                raise RuntimeError(
                    f"Native model preparation failed: {stdout.decode(errors='replace').strip()} {stderr.decode(errors='replace').strip()}"
                )
            if len(stdout) > 1_048_576:
                raise RuntimeError("Native preparation receipt exceeds 1 MiB")
            receipt = json.loads(stdout)
            if (
                not isinstance(receipt, dict)
                or receipt.get("type") != "prepared"
                or receipt.get("model_id") != model_id
            ):
                raise RuntimeError(
                    "Native model preparation returned no matching residency receipt"
                )
            return receipt
        finally:
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), 2.0)
                except TimeoutError:
                    with contextlib.suppress(ProcessLookupError):
                        process.kill()
                    await process.wait()

    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self._process = process
        self._write_lock = asyncio.Lock()
        self._next_id = 0
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._events: deque[dict[str, Any]] = deque()
        self._condition = asyncio.Condition()
        self._stderr: deque[str] = deque(maxlen=32)
        self._done = False
        self._closed = False
        self._shutdown: asyncio.Task[None] | None = None
        self._ready: asyncio.Future[dict[str, Any]] = (
            asyncio.get_running_loop().create_future()
        )
        self._reader = asyncio.create_task(
            self._read_events(), name="orchard-duplex-events"
        )
        self._errors = asyncio.create_task(
            self._read_stderr(), name="orchard-duplex-stderr"
        )
        self.epoch = 0
        self._frame_samples = FRAME_SAMPLES
        self._supports_grounded_response = False
        self._supports_response_hold = False

    @classmethod
    async def open(
        cls,
        model_id: str | None = None,
        *,
        binary: str | os.PathLike[str] | None = None,
        options: dict[str, Any] | None = None,
        ready_timeout: float = 300.0,
    ) -> Self:
        command = [
            _binary(binary, cls._program, cls._environment),
            "--model",
            model_id or cls._default_model,
        ]
        command.extend(
            [
                "--options-json",
                json.dumps(cls.effective_options(options), allow_nan=False),
            ]
        )
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=1_048_576,
        )
        session = cls(process)
        try:
            await asyncio.wait_for(asyncio.shield(session._ready), ready_timeout)
        except BaseException:
            await session.close()
            raise
        return session

    async def _read_stderr(self) -> None:
        assert self._process.stderr is not None
        async for line in self._process.stderr:
            self._stderr.append(line.decode("utf-8", errors="replace").rstrip())

    async def _read_events(self) -> None:
        assert self._process.stdout is not None
        failure: BaseException | None = None
        try:
            async for line in self._process.stdout:
                if self._done:
                    # Keep draining after overflow so the child's stdout cannot
                    # block its graceful close or process.wait().
                    continue
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise TypeError("Native audio event must be a JSON object")
                kind = event.get("type")
                request_id = event.get("id")
                if kind != "request_error" and "epoch" in event:
                    # Native timeline rollover has no command ACK. Even an ACK
                    # whose caller timed out must advance stale-output filtering.
                    self.epoch = max(self.epoch, int(event["epoch"]))
                if kind in {"audio_ack", "control_ack", "request_error"}:
                    future = self._pending.pop(request_id, None)
                    if future is not None and not future.done():
                        if kind == "request_error":
                            future.set_exception(
                                RuntimeError(
                                    event.get("message", "Duplex command failed")
                                )
                            )
                        else:
                            future.set_result(event)
                    continue
                if kind == "ready" and not self._ready.done():
                    self._frame_samples = int(event.get("frame_samples", FRAME_SAMPLES))
                    self._supports_grounded_response = (
                        event.get("supports_grounded_response") is True
                    )
                    self._supports_response_hold = (
                        event.get("supports_response_hold") is True
                    )
                    self._ready.set_result(event)
                elif kind == "error":
                    failure = RuntimeError(
                        event.get("message", "Duplex inference failed")
                    )
                    if not self._ready.done():
                        self._ready.set_exception(failure)
                async with self._condition:
                    # Keep the SDK side bounded too if playback pauses. Drop old
                    # PCM first, preserving speech/control/error information.
                    if len(self._events) >= 64:
                        old_audio = next(
                            (e for e in self._events if e.get("type") == "audio"), None
                        )
                        if old_audio is not None:
                            self._events.remove(old_audio)
                        else:
                            failure = RuntimeError(
                                "Native audio consumer is not draining its bounded event queue"
                            )
                            self._events.clear()
                            self._events.extend(
                                [
                                    {"type": "error", "message": str(failure)},
                                    {"type": "closed"},
                                ]
                            )
                            self._done = True
                            self._reject_pending(failure)
                            self._begin_close()
                            self._condition.notify_all()
                            continue
                    self._events.append(event)
                    self._condition.notify_all()
        except asyncio.CancelledError as exc:
            failure = exc
            raise
        except (OSError, ValueError, TypeError) as exc:
            failure = exc
        finally:
            if failure is None:
                failure = RuntimeError(
                    "Orchard duplex process closed"
                    + (": " + "\n".join(self._stderr) if self._stderr else "")
                )
            self._reject_pending(failure)
            async with self._condition:
                self._done = True
                self._condition.notify_all()

    def _reject_pending(self, failure: BaseException) -> None:
        if not self._ready.done():
            self._ready.set_exception(failure)
        for future in self._pending.values():
            if not future.done():
                future.set_exception(failure)
        self._pending.clear()

    async def _request(self, command: dict[str, Any]) -> dict[str, Any]:
        if self._done or self._closed:
            raise RuntimeError("Orchard duplex session is closed")
        assert self._process.stdin is not None
        async with self._write_lock:
            if self._done or self._closed:
                raise RuntimeError("Orchard duplex session is closed")
            request_id = self._next_id
            self._next_id += 1
            future: asyncio.Future[dict[str, Any]] = (
                asyncio.get_running_loop().create_future()
            )
            self._pending[request_id] = future
            try:
                payload = json.dumps(
                    {"id": request_id, **command},
                    allow_nan=False,
                    separators=(",", ":"),
                )
                self._process.stdin.write(payload.encode("utf-8") + b"\n")
                await self._process.stdin.drain()
            except BaseException:
                self._pending.pop(request_id, None)
                if future.done() and not future.cancelled():
                    future.exception()
                else:
                    future.cancel()
                raise
        try:
            return await asyncio.wait_for(future, 10.0)
        finally:
            self._pending.pop(request_id, None)

    async def push_audio(self, sequence: int, pcm: Iterable[float]) -> dict[str, Any]:
        samples = list(pcm)
        if len(samples) != self._frame_samples or any(
            not math.isfinite(s) or abs(s) > 1 for s in samples
        ):
            raise ValueError(
                f"Audio must contain {self._frame_samples} finite float32 samples in [-1, 1]"
            )
        if sequence < 0:
            raise ValueError("Audio sequence must be nonnegative")
        return await self._request(
            {"type": "audio", "sequence": sequence, "pcm": samples}
        )

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            async with self._condition:
                await self._condition.wait_for(lambda: bool(self._events) or self._done)
                if not self._events:
                    return
                event = self._events.popleft()
            if (
                event.get("type") in _EPOCH_EVENTS
                and int(event.get("epoch", 0)) < self.epoch
            ):
                continue
            yield event

    def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        return self.events()

    def _begin_close(self) -> asyncio.Task[None]:
        if self._shutdown is None:
            self._closed = True
            self._shutdown = asyncio.create_task(
                self._stop_process(), name="orchard-duplex-close"
            )
            # Automatic overflow shutdown may have no caller awaiting close.
            # Retrieving an exception here does not prevent close from raising it.
            self._shutdown.add_done_callback(
                lambda task: None if task.cancelled() else task.exception()
            )
        return self._shutdown

    async def _stop_process(self) -> None:
        async def close_input() -> None:
            async with self._write_lock:
                if (
                    self._process.stdin is not None
                    and not self._process.stdin.is_closing()
                ):
                    self._process.stdin.write(b'{"type":"close"}\n')
                    await self._process.stdin.drain()

        # A blocked writer must not prevent close from reaching termination.
        with contextlib.suppress(BrokenPipeError, ConnectionResetError, TimeoutError):
            await asyncio.wait_for(close_input(), 1.0)
        if self._process.stdin is not None:
            self._process.stdin.close()
        try:
            await asyncio.wait_for(self._process.wait(), 10.0)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), 5.0)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    self._process.kill()
                await self._process.wait()

    async def close(self) -> None:
        # Caller cancellation cannot cancel the one process cleanup task. The
        # reader never awaits this task, so automatic shutdown cannot self-join.
        await asyncio.shield(self._begin_close())
        await asyncio.gather(self._reader, self._errors, return_exceptions=True)
        if self._ready.done() and not self._ready.cancelled():
            self._ready.exception()  # Consume a startup error after a readiness timeout.

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()


class DuplexSession(_NativeAudioSession):
    """Native autonomous MoshiRAG speech with factual references and barge-in epochs.

    Default options advance only on supplied PCM. ``speak`` conditions the
    reference channel in this mode; the model chooses its own spoken wording.
    """

    @property
    def supports_grounded_response(self) -> bool:
        return self._supports_grounded_response

    @property
    def supports_response_hold(self) -> bool:
        return self._supports_response_hold

    async def _response_control(self, command: dict[str, Any]) -> int:
        expected_epoch = command["expected_epoch"]
        if type(expected_epoch) is not int or not 0 <= expected_epoch < (1 << 64):
            raise ValueError("Expected epoch must be an unsigned 64-bit integer")
        if expected_epoch != self.epoch:
            raise RuntimeError("Response control belongs to an interrupted epoch")
        receipt = await self._request(command)
        if (
            receipt.get("type") != "control_ack"
            or receipt.get("action") != command["type"]
            or type(receipt.get("epoch")) is not int
            or receipt["epoch"] != expected_epoch
            or self.epoch != expected_epoch
        ):
            raise RuntimeError(
                "Response control acknowledgement has a different action or epoch"
            )
        version = receipt.get("version")
        if type(version) is not int or not 0 < version < (1 << 64):
            raise RuntimeError(
                "Response control acknowledgement has no valid reference version"
            )
        return version

    async def hold_response(self, *, expected_epoch: int) -> int:
        """Request a response hold while native audio and listening continue.

        Returns the admitted reference version without advancing the epoch.
        The acknowledgement does not confirm that native sampling applied the hold.
        Interrupt and reset retire the hold; a grounded reply supplies its facts.
        """
        if not self.supports_response_hold:
            raise RuntimeError(
                "This native duplex session does not support response holds"
            )
        return await self._response_control(
            {"type": "hold_response", "expected_epoch": expected_epoch}
        )

    async def grounded_reply(self, text: str, *, expected_epoch: int) -> int:
        """Supply facts and request a response in the model's own words.

        Returns the queued reference version, not confirmation of playback.
        """
        if not self.supports_grounded_response:
            raise RuntimeError(
                "This native duplex session does not support grounded responses"
            )
        if not text.strip() or len(text.encode("utf-8")) > 8192:
            raise ValueError(
                "Grounded response context must contain 1..8192 UTF-8 bytes"
            )
        return await self._response_control(
            {"type": "grounded_reply", "text": text, "expected_epoch": expected_epoch}
        )

    async def speak(self, text: str, *, replace: bool = False) -> int:
        receipt = await self._request(
            {"type": "speak", "text": text, "replace": replace}
        )
        return int(receipt["epoch"])

    async def reference(self, text: str, *, expected_epoch: int) -> int:
        """Condition MoshiRAG on current facts; reject stale results after barge-in.

        Returns the reference version. Consume ``reference_applied`` before
        treating its content as available to the voice model. This is a trained
        context channel, so the model chooses its own spoken wording.
        """
        receipt = await self._request(
            {"type": "reference", "text": text, "expected_epoch": expected_epoch}
        )
        return int(receipt["version"])

    async def interrupt(self) -> int:
        receipt = await self._request({"type": "interrupt"})
        return int(receipt["epoch"])

    async def reset(self) -> int:
        receipt = await self._request({"type": "reset"})
        return int(receipt["epoch"])
