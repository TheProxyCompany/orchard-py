from __future__ import annotations

import asyncio
import json
import logging
import weakref
from collections.abc import Callable
from dataclasses import dataclass

import pynng

from proxy_inference_engine.engine.global_context import GlobalContext
from proxy_inference_engine.ipc.utils import ResponseDeltaDict

logger = logging.getLogger(__name__)

EVENT_TOPIC_PREFIX = b"__PIE_EVENT__:"


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
        self.management_lock = asyncio.Lock()

        self.response_channel_id: int = 0
        self.active_request_queues: dict[int, QueueRegistration] = {}

        self.request_id_counter: int = 0
        self.dispatcher_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self.response_topic_prefix: bytes = b""
        self.response_topic_prefix_len: int = 0

        self.global_context = weakref.ref(global_context)

    async def get_next_request_id(self) -> int:
        """Atomically increments and returns the next request ID."""
        async with self._lock:
            self.request_id_counter += 1
            # Basic overflow protection
            if self.request_id_counter >= 2**63:
                self.request_id_counter = 1
            return self.request_id_counter

    def handle_response_delta(self, msg_bytes: bytes) -> None:
        """Dispatches a response delta to the registered client queue."""
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
                f"Received delta for unknown/completed request_id {request_id}. Discarding."
            )

    def handle_engine_event(self, msg_bytes: bytes) -> None:
        """Handles engine events broadcasted from the engine."""
        parts = msg_bytes.split(b"\x00", 1)
        if len(parts) != 2:
            utf8_body = msg_bytes.decode("utf-8", errors="replace")
            logger.warning(f"Received malformed event message: {utf8_body}")
            return

        topic_part, json_body = parts
        event_name = topic_part[len(EVENT_TOPIC_PREFIX) :].decode("utf-8")
        ctx = self.global_context()
        try:
            payload = json.loads(json_body)
        except Exception as e:
            logger.error(f"Failed to parse engine event payload: {e!s}")
            return

        if event_name == "telemetry" and ctx is not None:
            ctx.last_telemetry = payload
            return

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

        logger.warning(f"Received unknown engine event '{event_name}'.")

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

        dispatcher = IPCDispatcher()
        resp_topic_prefix = f"resp:{ipc_state.response_channel_id:x}:".encode()
        ipc_state.response_topic_prefix = resp_topic_prefix
        ipc_state.response_topic_prefix_len = len(resp_topic_prefix)

        dispatcher.register_handler(resp_topic_prefix, IPCState.handle_response_delta)
        dispatcher.register_handler(EVENT_TOPIC_PREFIX, IPCState.handle_engine_event)

        try:
            while True:
                try:
                    msg = await sub_socket.arecv_msg()
                    if not dispatcher.dispatch(ipc_state, msg.bytes):
                        logger.warning("Received IPC message with unregistered prefix.")
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
            logger.info("Response dispatcher task finished.")
