from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import random
import sys
import threading
from array import array
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from dataclasses import dataclass
from typing import Any, Iterable, Literal, TypeVar, overload

from orchard.app.ipc_dispatch import IPCState, QueueRegistration
from orchard.app.model_registry import ModelRegistry
from orchard.clients.responses import (
    ResponseEvent,
    ResponsesRequest,
    aggregate_non_streaming_response,
    iter_response_events,
)
from orchard.defaults import MAX_GENERATED_TOKENS
from orchard.engine import ClientDelta, ClientResponse, UsageStats
from orchard.formatter import ChatFormatter
from orchard.formatter.multimodal import (
    build_multimodal_layout,
    build_multimodal_messages,
)
from orchard.ipc.serialization import _build_request_payload
from orchard.ipc.utils import (
    ResponseDeltaDict,
    normalise_delta_payload,
    release_delta_resources,
)
from orchard.server.exceptions import InferenceError
from orchard.server.models.reasoning import DEFAULT_BOOLEAN_REASONING_EFFORT
from orchard.server.models.responses import OutputTextDeltaEvent, ResponseObject

logger = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass(frozen=True)
class ModalArtifact:
    type: str
    event: str
    mime_type: str
    decoder_id: str
    data: bytes
    metadata: dict[str, Any] | None
    deltas: list[ClientDelta]


class AudioClient:
    def __init__(self, client: Client):
        self._client = client

    async def agenerate(
        self,
        model_id: str,
        text: str,
        *,
        reference_audio: bytes | bytearray | memoryview | None = None,
        language: str | None = None,
        speaker: str | None = None,
        sample_rate: int = 24000,
        max_output_tokens: int = 8192,
        stream: bool = False,
        **options: Any,
    ) -> list[ModalArtifact] | AsyncIterator[ClientDelta]:
        modal_options = {
            "language": language,
            "speaker": speaker,
            "sample_rate": sample_rate,
            **options,
        }
        audio_buffers = [bytes(reference_audio)] if reference_audio is not None else []
        return await self._client._amodal_artifacts(
            request_type="audio",
            model_id=model_id,
            prompt=text,
            task_name="text_to_speech",
            modal_options=modal_options,
            audio_buffers=audio_buffers,
            max_generated_tokens=max_output_tokens,
            stream=stream,
        )

    def generate(
        self,
        model_id: str,
        text: str,
        *,
        reference_audio: bytes | bytearray | memoryview | None = None,
        language: str | None = None,
        speaker: str | None = None,
        sample_rate: int = 24000,
        max_output_tokens: int = 8192,
        stream: bool = False,
        **options: Any,
    ) -> list[ModalArtifact] | Iterator[ClientDelta]:
        result = self._client._sync_submit(
            self.agenerate(
                model_id,
                text,
                reference_audio=reference_audio,
                language=language,
                speaker=speaker,
                sample_rate=sample_rate,
                max_output_tokens=max_output_tokens,
                stream=stream,
                **options,
            )
        )
        if isinstance(result, AsyncIterator):
            return self._client._sync_iterator_bridge(result)
        return result

    async def asynthesize(
        self, *args: Any, **kwargs: Any
    ) -> list[ModalArtifact] | AsyncIterator[ClientDelta]:
        return await self.agenerate(*args, **kwargs)

    def synthesize(
        self, *args: Any, **kwargs: Any
    ) -> list[ModalArtifact] | Iterator[ClientDelta]:
        return self.generate(*args, **kwargs)

    async def atranscribe(
        self,
        model_id: str,
        pcm: Iterable[float],
    ) -> str:
        return await self._client._atranscribe_audio(model_id, pcm)

    def transcribe(self, model_id: str, pcm: Iterable[float]) -> str:
        return self._client._sync_submit(self.atranscribe(model_id, pcm))


class ImagesClient:
    def __init__(self, client: Client):
        self._client = client

    async def agenerate(
        self,
        model_id: str,
        prompt: str,
        *,
        height: int = 1024,
        width: int = 1024,
        num_steps: int = 48,
        guidance_scale: float = 7.0,
        seed: int | None = None,
        stream: bool = False,
        **options: Any,
    ) -> list[ModalArtifact] | AsyncIterator[ClientDelta]:
        modal_options = {
            "height": height,
            "width": width,
            "num_steps": num_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            **options,
        }
        return await self._client._amodal_artifacts(
            request_type="image",
            model_id=model_id,
            prompt=prompt,
            task_name="text_to_image",
            modal_options=modal_options,
            max_generated_tokens=0,
            stream=stream,
        )

    def generate(
        self,
        model_id: str,
        prompt: str,
        *,
        height: int = 1024,
        width: int = 1024,
        num_steps: int = 48,
        guidance_scale: float = 7.0,
        seed: int | None = None,
        stream: bool = False,
        **options: Any,
    ) -> list[ModalArtifact] | Iterator[ClientDelta]:
        result = self._client._sync_submit(
            self.agenerate(
                model_id,
                prompt,
                height=height,
                width=width,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                stream=stream,
                **options,
            )
        )
        if isinstance(result, AsyncIterator):
            return self._client._sync_iterator_bridge(result)
        return result

    async def aedit(
        self,
        model_id: str,
        image: bytes | bytearray | memoryview,
        prompt: str,
        *,
        height: int | None = None,
        width: int | None = None,
        num_steps: int = 50,
        true_cfg_scale: float = 4.0,
        negative_prompt: str = " ",
        seed: int | None = None,
        stream: bool = False,
        **options: Any,
    ) -> list[ModalArtifact] | AsyncIterator[ClientDelta]:
        modal_options = {
            "height": height,
            "width": width,
            "num_steps": num_steps,
            "true_cfg_scale": true_cfg_scale,
            "negative_prompt": negative_prompt,
            "seed": seed,
            **options,
        }
        return await self._client._amodal_artifacts(
            request_type="image",
            model_id=model_id,
            prompt=prompt,
            task_name="image_to_image",
            modal_options=modal_options,
            max_generated_tokens=0,
            stream=stream,
            image_buffers=[bytes(image)],
        )

    def edit(
        self,
        model_id: str,
        image: bytes | bytearray | memoryview,
        prompt: str,
        *,
        height: int | None = None,
        width: int | None = None,
        num_steps: int = 50,
        true_cfg_scale: float = 4.0,
        negative_prompt: str = " ",
        seed: int | None = None,
        stream: bool = False,
        **options: Any,
    ) -> list[ModalArtifact] | Iterator[ClientDelta]:
        result = self._client._sync_submit(
            self.aedit(
                model_id,
                image,
                prompt,
                height=height,
                width=width,
                num_steps=num_steps,
                true_cfg_scale=true_cfg_scale,
                negative_prompt=negative_prompt,
                seed=seed,
                stream=stream,
                **options,
            )
        )
        if isinstance(result, AsyncIterator):
            return self._client._sync_iterator_bridge(result)
        return result


class Client:
    """
    A high-level Python client for the Proxy Inference Engine (PIE).

    Provides both synchronous and asynchronous interfaces for interacting
    with the shared, on-demand engine service managed by InferenceEngine.
    """

    def __init__(self, ipc_state: IPCState, model_registry: ModelRegistry):
        self._ipc_state = ipc_state
        self._model_registry = model_registry
        self.audio = AudioClient(self)
        self.images = ImagesClient(self)

        # For the synchronous wrapper
        self._sync_loop: asyncio.AbstractEventLoop | None = None
        self._sync_thread: threading.Thread | None = None
        self._sync_start_lock = threading.Lock()

    def resolve_capabilities(self, model_id: str) -> dict[str, int]:
        """Resolve control token capabilities for a model into token IDs."""
        info = self._model_registry.ensure_ready_sync(model_id)
        capabilities = info.capabilities or {}
        resolved: dict[str, int] = {}
        for name, token_ids in capabilities.items():
            if isinstance(token_ids, list | tuple) and token_ids:
                resolved[name] = int(token_ids[0])
            elif isinstance(token_ids, int | float):
                resolved[name] = int(token_ids)
            else:
                raise TypeError(f"Unsupported capability token format for '{name}'.")
        return resolved

    async def acancel_request(self, request_id: int) -> dict[str, Any]:
        """Cancel an in-flight PIE request by request id."""
        return await self._ipc_state.cancel_request(request_id)

    def cancel_request(self, request_id: int) -> dict[str, Any]:
        """Synchronous wrapper for canceling an in-flight PIE request."""
        if (
            not self._sync_loop
            or not self._sync_thread
            or not self._sync_thread.is_alive()
        ):
            self._start_sync_event_loop()

        assert self._sync_loop, "Sync loop not initialized"
        future = asyncio.run_coroutine_threadsafe(
            self.acancel_request(request_id),
            self._sync_loop,
        )
        return future.result()

    async def acancel_model_load(self, model_id: str) -> dict | None:
        """Cancel an in-progress PIE model load or activation."""
        return await self._model_registry.cancel_activation(model_id)

    def cancel_model_load(self, model_id: str) -> dict | None:
        """Synchronous wrapper for canceling an in-progress model load."""
        if (
            not self._sync_loop
            or not self._sync_thread
            or not self._sync_thread.is_alive()
        ):
            self._start_sync_event_loop()

        assert self._sync_loop, "Sync loop not initialized"
        future = asyncio.run_coroutine_threadsafe(
            self.acancel_model_load(model_id),
            self._sync_loop,
        )
        return future.result()

    async def _cancel_request_for_cleanup(self, request_id: int) -> None:
        try:
            await self.acancel_request(request_id)
        except Exception:
            logger.debug(
                "Failed to cancel interrupted PIE request %d.",
                request_id,
                exc_info=True,
            )

    def _sync_submit(self, coro: Any) -> Any:
        if (
            not self._sync_loop
            or not self._sync_thread
            or not self._sync_thread.is_alive()
        ):
            self._start_sync_event_loop()

        assert self._sync_loop, "Sync loop not initialized"
        future = asyncio.run_coroutine_threadsafe(coro, self._sync_loop)
        return future.result()

    async def _amodal_artifacts(
        self,
        *,
        request_type: str,
        model_id: str,
        prompt: str,
        task_name: str,
        modal_options: dict[str, Any],
        max_generated_tokens: int,
        stream: bool,
        image_buffers: list[bytes] | None = None,
        audio_buffers: list[bytes] | None = None,
    ) -> list[ModalArtifact] | AsyncIterator[ClientDelta]:
        if self._ipc_state.engine_dead:
            raise RuntimeError("Engine process is dead; cannot submit new requests.")

        request_id = await self._ipc_state.get_next_request_id()
        response_queue: asyncio.Queue[ResponseDeltaDict] = asyncio.Queue()
        owner_loop = asyncio.get_running_loop()
        self._ipc_state.active_request_queues[request_id] = QueueRegistration(
            loop=owner_loop, queue=response_queue
        )
        queue_cleared = False
        stream_managed_cleanup = False

        def _cleanup_queue() -> None:
            nonlocal queue_cleared
            if queue_cleared:
                return
            queue_cleared = True
            self._ipc_state.active_request_queues.pop(request_id, None)

        try:
            info = await self._model_registry.get_info(model_id)
            response_channel_id = self._ipc_state.response_channel_id or request_id
            prompt_bytes = prompt.encode("utf-8")
            image_buffers = image_buffers or []
            audio_buffers = audio_buffers or []
            layout: list[dict[str, int | str]] = []
            if prompt_bytes or (not image_buffers and not audio_buffers):
                layout.append({"type": "text", "length": len(prompt_bytes)})
            layout.extend(
                {"type": "image", "length": len(buffer)} for buffer in image_buffers
            )
            layout.extend(
                {"type": "audio", "length": len(buffer)} for buffer in audio_buffers
            )
            clean_options = {
                key: value for key, value in modal_options.items() if value is not None
            }
            modal_options_json = json.dumps(
                clean_options,
                separators=(",", ":"),
                ensure_ascii=False,
                sort_keys=True,
            )
            request_bytes = _build_request_payload(
                request_id=request_id,
                model_id=info.model_id,
                model_path=info.model_path,
                request_type=request_type,
                response_channel_id=response_channel_id,
                prompts=[
                    {
                        "prompt_bytes": prompt_bytes,
                        "image_buffers": image_buffers,
                        "audio_buffers": audio_buffers,
                        "layout": layout,
                        "max_generated_tokens": max_generated_tokens,
                        "task_name": task_name,
                        "modal_options_json": modal_options_json,
                        "sampling_params": {
                            "temperature": float(clean_options.get("temperature", 1.0)),
                            "top_p": float(clean_options.get("top_p", 1.0)),
                            "top_k": int(clean_options.get("top_k", -1)),
                            "min_p": float(clean_options.get("min_p", 0.0)),
                            "rng_seed": int(clean_options.get("seed", 0)),
                            "deterministic": bool(
                                clean_options.get("deterministic", False)
                            ),
                        },
                    }
                ],
            )
            socket = self._ipc_state.request_socket
            if socket is None:
                raise RuntimeError("Request socket is not initialized.")
            await socket.asend(request_bytes)

            async def _stream_generator() -> AsyncIterator[ClientDelta]:
                stream_processor = self._async_process_stream(
                    response_queue,
                    on_cancel=lambda: self._cancel_request_for_cleanup(request_id),
                )
                try:
                    async for delta in stream_processor:
                        yield delta
                finally:
                    await stream_processor.aclose()
                    _cleanup_queue()

            if stream:
                stream_managed_cleanup = True
                return _stream_generator()

            deltas = [
                delta
                async for delta in self._async_process_stream(
                    response_queue,
                    on_cancel=lambda: self._cancel_request_for_cleanup(request_id),
                )
            ]
            _cleanup_queue()
            return self._modal_artifacts_from_deltas(deltas)
        finally:
            if not stream or not stream_managed_cleanup:
                _cleanup_queue()

    @staticmethod
    def _modal_artifacts_from_deltas(deltas: list[ClientDelta]) -> list[ModalArtifact]:
        error_message = next(
            (
                delta.error_message or delta.content or "Modal artifact request failed."
                for delta in deltas
                if delta.error_message or (delta.finish_reason or "").lower() == "error"
            ),
            None,
        )
        if error_message is not None:
            raise InferenceError(error_message)

        artifacts: list[ModalArtifact] = []
        for delta in deltas:
            if not delta.modal_bytes_b64:
                continue
            metadata = None
            if delta.modal_metadata_json:
                metadata = json.loads(delta.modal_metadata_json)
            artifacts.append(
                ModalArtifact(
                    type=delta.modal_type or "",
                    event=delta.modal_event or "",
                    mime_type=delta.modal_mime_type or "",
                    decoder_id=delta.modal_decoder_id or "",
                    data=base64.b64decode(delta.modal_bytes_b64),
                    metadata=metadata,
                    deltas=deltas,
                )
            )
        return artifacts

    async def _atranscribe_audio(self, model_id: str, pcm: Iterable[float]) -> str:
        pcm_bytes = self._encode_float32_pcm_bytes(pcm)
        if not pcm_bytes:
            return ""
        if self._ipc_state.engine_dead:
            raise RuntimeError("Engine process is dead; cannot submit new requests.")

        request_id = await self._ipc_state.get_next_request_id()
        response_queue: asyncio.Queue[ResponseDeltaDict] = asyncio.Queue()
        owner_loop = asyncio.get_running_loop()
        self._ipc_state.active_request_queues[request_id] = QueueRegistration(
            loop=owner_loop, queue=response_queue
        )
        queue_cleared = False

        def _cleanup_queue() -> None:
            nonlocal queue_cleared
            if queue_cleared:
                return
            queue_cleared = True
            self._ipc_state.active_request_queues.pop(request_id, None)

        try:
            info = await self._model_registry.get_info(model_id)
            response_channel_id = self._ipc_state.response_channel_id or request_id
            request_bytes = _build_request_payload(
                request_id=request_id,
                model_id=info.model_id,
                model_path=info.model_path,
                request_type="omni",
                response_channel_id=response_channel_id,
                prompts=[
                    {
                        "prompt_bytes": b"",
                        "audio_buffers": [pcm_bytes],
                        "layout": [
                            {"type": "text", "length": 0},
                            {"type": "audio", "length": len(pcm_bytes)},
                        ],
                        "max_generated_tokens": 0,
                    }
                ],
            )
            socket = self._ipc_state.request_socket
            if socket is None:
                raise RuntimeError("Request socket is not initialized.")
            await socket.asend(request_bytes)
            deltas = [
                delta
                async for delta in self._async_process_stream(
                    response_queue,
                    on_cancel=lambda: self._cancel_request_for_cleanup(request_id),
                )
            ]
            error_message = next(
                (
                    delta.error_message
                    or delta.content
                    or "Transcription request failed."
                    for delta in deltas
                    if delta.error_message
                    or (delta.finish_reason or "").lower() == "error"
                ),
                None,
            )
            if error_message is not None:
                raise InferenceError(error_message)
            return "".join(delta.content or "" for delta in deltas)
        finally:
            _cleanup_queue()

    @staticmethod
    def _encode_float32_pcm_bytes(pcm: Iterable[float]) -> bytes:
        values = array("f", (float(sample) for sample in pcm))
        if values.itemsize != 4:
            raise RuntimeError("Platform float array is not 32-bit.")
        if sys.byteorder != "little":
            values.byteswap()
        return values.tobytes()

    async def _async_process_stream(
        self,
        response_queue: asyncio.Queue[ResponseDeltaDict],
        expected_final_prompt_count: int = 1,
        on_cancel: Callable[[], Awaitable[None]] | None = None,
    ) -> AsyncIterator[ClientDelta]:
        """The core async stream processor."""
        completed_prompt_indexes: set[int] = set()
        completed = False
        try:
            while True:
                delta = await response_queue.get()

                sanitized_delta = dict(delta)
                # Clear SHM-related fields before yielding
                for key in (
                    "bulk_content_view",
                    "bulk_content_bytes",
                    "embedding_bytes",
                ):
                    sanitized_delta.pop(key, None)

                client_delta = ClientDelta.model_validate(sanitized_delta)
                should_stop = False
                if client_delta.is_final:
                    if expected_final_prompt_count <= 1:
                        should_stop = True
                    elif client_delta.prompt_index is None:
                        should_stop = True
                    else:
                        completed_prompt_indexes.add(client_delta.prompt_index)
                        should_stop = (
                            len(completed_prompt_indexes) >= expected_final_prompt_count
                        )

                if should_stop:
                    completed = True

                try:
                    yield client_delta
                finally:
                    release_delta_resources(
                        delta
                    )  # Release resources after consumer is done

                if should_stop:
                    break
        finally:
            if not completed and on_cancel:
                await on_cancel()
            # Clean up any remaining items if the generator is exited early
            while not response_queue.empty():
                try:
                    leftover = response_queue.get_nowait()
                    release_delta_resources(leftover)
                except asyncio.QueueEmpty:
                    break

    async def _async_process_raw_stream(
        self,
        response_queue: asyncio.Queue[ResponseDeltaDict],
        on_cancel: Callable[[], Awaitable[None]] | None = None,
    ) -> AsyncIterator[ResponseDeltaDict]:
        """Yield raw response deltas for Responses API event mapping."""
        # Intentionally mirrors _async_process_stream, but yields raw payloads
        # (including state_events) instead of ClientDelta objects.
        completed = False
        try:
            while True:
                delta = await response_queue.get()
                normalise_delta_payload(delta)
                should_stop = bool(delta.get("is_final_delta"))
                if should_stop:
                    completed = True

                try:
                    yield delta
                finally:
                    release_delta_resources(delta)

                if should_stop:
                    break
        finally:
            if not completed and on_cancel:
                await on_cancel()
            while not response_queue.empty():
                try:
                    leftover = response_queue.get_nowait()
                    release_delta_resources(leftover)
                except asyncio.QueueEmpty:
                    break

    async def _asubmit_prefill_task(
        self,
        request_id: int,
        model_id: str,
        texts: list[str],
        task_name: str,
    ) -> None:
        info = await self._model_registry.get_info(model_id)
        engine_model_id = info.model_id
        response_channel_id = self._ipc_state.response_channel_id or request_id
        request_bytes = _build_request_payload(
            request_id=request_id,
            model_id=engine_model_id,
            model_path=info.model_path,
            request_type="prefill_task",
            response_channel_id=response_channel_id,
            prompts=[
                {
                    "prompt_bytes": text.encode("utf-8"),
                    "max_generated_tokens": 0,
                    "task_name": task_name,
                }
                for text in texts
            ],
        )
        socket = self._ipc_state.request_socket
        if socket is None:
            raise RuntimeError("Request socket is not initialized.")
        await socket.asend(request_bytes)

    async def aprefill_task(
        self,
        model_id: str,
        text: str,
        task_name: str,
        *,
        stream: bool = False,
    ) -> list[ClientDelta] | AsyncIterator[ClientDelta]:
        if self._ipc_state.engine_dead:
            raise RuntimeError("Engine process is dead; cannot submit new requests.")

        request_id = await self._ipc_state.get_next_request_id()
        response_queue: asyncio.Queue[ResponseDeltaDict] = asyncio.Queue()
        owner_loop = asyncio.get_running_loop()
        self._ipc_state.active_request_queues[request_id] = QueueRegistration(
            loop=owner_loop, queue=response_queue
        )
        queue_cleared = False
        stream_managed_cleanup = False

        def _cleanup_queue() -> None:
            nonlocal queue_cleared
            if queue_cleared:
                return
            queue_cleared = True
            self._ipc_state.active_request_queues.pop(request_id, None)

        try:
            await self._asubmit_prefill_task(request_id, model_id, [text], task_name)

            async def _stream_generator() -> AsyncIterator[ClientDelta]:
                stream_processor = self._async_process_stream(
                    response_queue,
                    on_cancel=lambda: self._cancel_request_for_cleanup(request_id),
                )
                try:
                    async for delta in stream_processor:
                        yield delta
                finally:
                    await stream_processor.aclose()
                    _cleanup_queue()

            if stream:
                stream_managed_cleanup = True
                return _stream_generator()

            deltas = [
                delta
                async for delta in self._async_process_stream(
                    response_queue,
                    on_cancel=lambda: self._cancel_request_for_cleanup(request_id),
                )
            ]
            _cleanup_queue()
            return deltas
        finally:
            if not stream or not stream_managed_cleanup:
                _cleanup_queue()

    def prefill_task(
        self,
        model_id: str,
        text: str,
        task_name: str,
        *,
        stream: bool = False,
    ) -> list[ClientDelta] | Iterator[ClientDelta]:
        if (
            not self._sync_loop
            or not self._sync_thread
            or not self._sync_thread.is_alive()
        ):
            self._start_sync_event_loop()

        assert self._sync_loop, "Sync loop not initialized"
        future = asyncio.run_coroutine_threadsafe(
            self.aprefill_task(model_id, text, task_name, stream=stream),
            self._sync_loop,
        )
        result = future.result()
        if isinstance(result, AsyncIterator):
            return self._sync_iterator_bridge(result)
        return result

    async def aprefill_task_batch(
        self,
        model_id: str,
        texts: list[str],
        task_name: str,
    ) -> list[list[ClientDelta]]:
        if not texts:
            return []
        if self._ipc_state.engine_dead:
            raise RuntimeError("Engine process is dead; cannot submit new requests.")

        request_id = await self._ipc_state.get_next_request_id()
        response_queue: asyncio.Queue[ResponseDeltaDict] = asyncio.Queue()
        owner_loop = asyncio.get_running_loop()
        self._ipc_state.active_request_queues[request_id] = QueueRegistration(
            loop=owner_loop, queue=response_queue
        )

        try:
            await self._asubmit_prefill_task(request_id, model_id, texts, task_name)
            deltas = [
                delta
                async for delta in self._async_process_stream(
                    response_queue,
                    expected_final_prompt_count=len(texts),
                    on_cancel=lambda: self._cancel_request_for_cleanup(request_id),
                )
            ]
        finally:
            self._ipc_state.active_request_queues.pop(request_id, None)

        deltas_by_prompt: list[list[ClientDelta]] = [[] for _ in texts]
        for delta in deltas:
            prompt_index = delta.prompt_index if delta.prompt_index is not None else 0
            if 0 <= prompt_index < len(deltas_by_prompt):
                deltas_by_prompt[prompt_index].append(delta)
        return deltas_by_prompt

    def prefill_task_batch(
        self,
        model_id: str,
        texts: list[str],
        task_name: str,
    ) -> list[list[ClientDelta]]:
        if (
            not self._sync_loop
            or not self._sync_thread
            or not self._sync_thread.is_alive()
        ):
            self._start_sync_event_loop()

        assert self._sync_loop, "Sync loop not initialized"
        future = asyncio.run_coroutine_threadsafe(
            self.aprefill_task_batch(model_id, texts, task_name),
            self._sync_loop,
        )
        return future.result()

    def _build_responses_request(
        self,
        *,
        request: ResponsesRequest | None,
        input: str | list[dict[str, Any]] | None,
        stream: bool,
        **kwargs: Any,
    ) -> ResponsesRequest:
        if request is not None:
            return request
        if input is None:
            raise ValueError("Responses request requires either 'input' or 'request'.")

        request_payload: dict[str, Any] = {"input": input, "stream": stream}
        request_payload.update(kwargs)
        filtered_payload = {
            key: value for key, value in request_payload.items() if value is not None
        }
        return ResponsesRequest(**filtered_payload)

    @staticmethod
    def _is_batched_messages(messages: list) -> bool:
        """Check if messages is a batch (list of conversations) or single conversation."""
        if not messages:
            return False
        # Batched: [[{role, content}, ...], [{role, content}, ...]]
        # Single: [{role, content}, ...]
        return isinstance(messages[0], list)

    @staticmethod
    def _normalize_messages(messages: list) -> list[list[dict]]:
        """Normalize messages to always be a list of conversations."""
        if Client._is_batched_messages(messages):
            return messages
        return [messages]

    @staticmethod
    def _kwargs_for_prompt(
        kwargs: dict[str, Any], prompt_index: int, batch_size: int
    ) -> dict[str, Any]:
        prompt_kwargs = dict(kwargs)
        response_format = kwargs.get("response_format")
        if isinstance(response_format, list):
            if len(response_format) == batch_size:
                prompt_kwargs["response_format"] = response_format[prompt_index]
            elif len(response_format) == 1:
                prompt_kwargs["response_format"] = response_format[0]
            else:
                raise ValueError(
                    f"Length of 'response_format' ({len(response_format)}) does not match batch size {batch_size}."
                )
        return prompt_kwargs

    async def arender_prompt(
        self,
        model_id: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Render a single chat prompt exactly as Orchard will submit it."""
        info = await self._model_registry.get_info(model_id)
        _, capture_payload = self._prepare_prompt_payload(
            model_id=model_id,
            model_path=info.model_path,
            formatter=info.formatter,
            messages=messages,
            **kwargs,
        )
        return capture_payload

    def render_prompt(
        self,
        model_id: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Synchronous wrapper for `arender_prompt()`."""
        if (
            not self._sync_loop
            or not self._sync_thread
            or not self._sync_thread.is_alive()
        ):
            self._start_sync_event_loop()

        assert self._sync_loop, "Sync loop not initialized"
        future = asyncio.run_coroutine_threadsafe(
            self.arender_prompt(model_id, messages, **kwargs),
            self._sync_loop,
        )
        return future.result()

    async def arender_responses_prompt(
        self,
        model_id: str,
        *,
        input: str | list[dict[str, Any]] | None = None,
        request: ResponsesRequest | None = None,
        stream: bool = False,
        instructions: str | None = None,
        tools: list[Any] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        deterministic: bool | None = None,
        max_output_tokens: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        top_logprobs: int | None = None,
        reasoning: Any | None = None,
        metadata: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Render a Responses API prompt exactly as Orchard will submit it."""
        request_kwargs: dict[str, Any] = {
            "instructions": instructions,
            "tools": tools,
            "tool_choice": tool_choice,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "deterministic": deterministic,
            "max_output_tokens": max_output_tokens,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "top_logprobs": top_logprobs,
            "reasoning": reasoning,
            "metadata": metadata,
        }
        request_kwargs.update(kwargs)
        response_request = self._build_responses_request(
            request=request,
            input=input,
            stream=stream,
            **request_kwargs,
        )
        return await self.arender_prompt(
            model_id,
            response_request.to_messages(),
            **response_request.to_submit_kwargs(),
        )

    def render_responses_prompt(
        self,
        model_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Synchronous wrapper for `arender_responses_prompt()`."""
        if (
            not self._sync_loop
            or not self._sync_thread
            or not self._sync_thread.is_alive()
        ):
            self._start_sync_event_loop()

        assert self._sync_loop, "Sync loop not initialized"
        future = asyncio.run_coroutine_threadsafe(
            self.arender_responses_prompt(model_id, **kwargs),
            self._sync_loop,
        )
        return future.result()

    async def achat(
        self,
        model_id: str,
        messages: list[dict] | list[list[dict]],
        stream: bool = False,
        **kwargs: Any,
    ) -> ClientResponse | list[ClientResponse] | AsyncIterator[ClientDelta]:
        """
        Asynchronously performs a chat completion.

        Args:
            model_id: The model to use for generation.
            messages: Either a single conversation (list of message dicts) or
                     a batch of conversations (list of list of message dicts).
            stream: If True, returns an async iterator of deltas.
            **kwargs: Additional generation parameters.

        Returns:
            - Single conversation, non-streaming: ClientResponse
            - Batched conversations, non-streaming: list[ClientResponse]
            - Streaming (single or batched): AsyncIterator[ClientDelta]
              (use delta.prompt_index to demultiplex batched streams)
        """
        if self._ipc_state.engine_dead:
            raise RuntimeError("Engine process is dead; cannot submit new requests.")

        is_batched = self._is_batched_messages(messages)
        conversations = self._normalize_messages(messages)
        batch_size = len(conversations)

        request_id = await self._ipc_state.get_next_request_id()
        response_queue: asyncio.Queue[ResponseDeltaDict] = asyncio.Queue()
        owner_loop = asyncio.get_running_loop()
        self._ipc_state.active_request_queues[request_id] = QueueRegistration(
            loop=owner_loop, queue=response_queue
        )
        queue_cleared = False
        stream_managed_cleanup = False

        def _cleanup_queue() -> None:
            nonlocal queue_cleared
            if queue_cleared:
                return
            queue_cleared = True
            self._ipc_state.active_request_queues.pop(request_id, None)

        try:
            await self._asubmit_request_batch(
                request_id, model_id, conversations, **kwargs
            )

            async def _stream_generator() -> AsyncIterator[ClientDelta]:
                stream_processor = self._async_process_stream(
                    response_queue,
                    on_cancel=lambda: self._cancel_request_for_cleanup(request_id),
                )
                try:
                    async for delta in stream_processor:
                        yield delta
                finally:
                    await stream_processor.aclose()
                    _cleanup_queue()

            if stream:
                stream_managed_cleanup = True
                return _stream_generator()
            else:
                deltas = [
                    delta
                    async for delta in self._async_process_stream(
                        response_queue,
                        batch_size if is_batched else 1,
                        on_cancel=lambda: self._cancel_request_for_cleanup(request_id),
                    )
                ]
                _cleanup_queue()
                responses = self._aggregate_batch_response(deltas, batch_size)
                return responses if is_batched else responses[0]
        finally:
            if not stream or not stream_managed_cleanup:
                _cleanup_queue()

    @overload
    async def aresponses(
        self,
        model_id: str,
        *,
        input: str | list[dict[str, Any]] | None = None,
        request: ResponsesRequest | None = None,
        stream: Literal[False] = False,
        instructions: str | None = None,
        tools: list[Any] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        max_output_tokens: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        top_logprobs: int | None = None,
        reasoning: Any | None = None,
        metadata: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> ResponseObject: ...

    @overload
    async def aresponses(
        self,
        model_id: str,
        *,
        input: str | list[dict[str, Any]] | None = None,
        request: ResponsesRequest | None = None,
        stream: Literal[True],
        instructions: str | None = None,
        tools: list[Any] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        max_output_tokens: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        top_logprobs: int | None = None,
        reasoning: Any | None = None,
        metadata: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ResponseEvent]: ...

    async def aresponses(
        self,
        model_id: str,
        *,
        input: str | list[dict[str, Any]] | None = None,
        request: ResponsesRequest | None = None,
        stream: bool = False,
        instructions: str | None = None,
        tools: list[Any] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        max_output_tokens: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        top_logprobs: int | None = None,
        reasoning: Any | None = None,
        metadata: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> ResponseObject | AsyncIterator[ResponseEvent]:
        """Run a Responses API request over IPC.

        Args:
            model_id: Model ID to execute against.
            input: String or item-list input (common path).
            request: Optional pre-built `ResponsesRequest` for power users.
            stream: When True, returns a typed event stream.
            kwargs: Additional Responses fields (for example `text`, `max_tool_calls`).
        Returns:
            `ResponseObject` for non-streaming calls, or `AsyncIterator[ResponseEvent]`.
        Example:
            `resp = await client.aresponses("llama3", input="hi")`
            `events = await client.aresponses("llama3", input="hi", stream=True)`
        """
        if self._ipc_state.engine_dead:
            raise RuntimeError("Engine process is dead; cannot submit new requests.")

        request_kwargs: dict[str, Any] = {
            "instructions": instructions,
            "tools": tools,
            "tool_choice": tool_choice,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "max_output_tokens": max_output_tokens,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "top_logprobs": top_logprobs,
            "reasoning": reasoning,
            "metadata": metadata,
        }
        request_kwargs.update(kwargs)
        response_request = self._build_responses_request(
            request=request,
            input=input,
            stream=stream,
            **request_kwargs,
        )

        request_id = await self._ipc_state.get_next_request_id()
        response_queue: asyncio.Queue[ResponseDeltaDict] = asyncio.Queue()
        owner_loop = asyncio.get_running_loop()
        self._ipc_state.active_request_queues[request_id] = QueueRegistration(
            loop=owner_loop, queue=response_queue
        )
        queue_cleared = False
        stream_managed_cleanup = False

        def _cleanup_queue() -> None:
            nonlocal queue_cleared
            if queue_cleared:
                return
            queue_cleared = True
            self._ipc_state.active_request_queues.pop(request_id, None)

        try:
            await self._asubmit_request(
                request_id,
                model_id,
                response_request.to_messages(),
                **response_request.to_submit_kwargs(),
            )

            async def _response_stream() -> AsyncIterator[ResponseEvent]:
                raw_stream = self._async_process_raw_stream(
                    response_queue,
                    on_cancel=lambda: self._cancel_request_for_cleanup(request_id),
                )
                event_stream = iter_response_events(
                    raw_stream,
                    model_id=model_id,
                    stream_tokens=response_request.stream_tokens,
                )
                try:
                    async for event in event_stream:
                        yield event
                finally:
                    with contextlib.suppress(Exception):
                        await event_stream.aclose()
                    await raw_stream.aclose()
                    _cleanup_queue()

            if response_request.stream:
                stream_managed_cleanup = True
                return _response_stream()

            deltas = [
                delta
                async for delta in self._async_process_raw_stream(
                    response_queue,
                    on_cancel=lambda: self._cancel_request_for_cleanup(request_id),
                )
            ]
            _cleanup_queue()
            return aggregate_non_streaming_response(deltas, model_id, response_request)
        finally:
            if not response_request.stream or not stream_managed_cleanup:
                _cleanup_queue()

    def chat(
        self,
        model_id: str,
        messages: list[dict] | list[list[dict]],
        stream: bool = False,
        **kwargs: Any,
    ) -> ClientResponse | list[ClientResponse] | Iterator[ClientDelta]:
        """
        Synchronously performs a chat completion.

        Args:
            model_id: The model to use for generation.
            messages: Either a single conversation (list of message dicts) or
                     a batch of conversations (list of list of message dicts).
            stream: If True, returns an iterator of deltas.
            **kwargs: Additional generation parameters.

        Returns:
            - Single conversation, non-streaming: ClientResponse
            - Batched conversations, non-streaming: list[ClientResponse]
            - Streaming (single or batched): Iterator[ClientDelta]
              (use delta.prompt_index to demultiplex batched streams)
        """
        # We need a running event loop in a background thread
        if (
            not self._sync_loop
            or not self._sync_thread
            or not self._sync_thread.is_alive()
        ):
            self._start_sync_event_loop()

        assert self._sync_loop, "Sync loop not initialized"
        future = asyncio.run_coroutine_threadsafe(
            self.achat(model_id, messages, stream=stream, **kwargs),
            self._sync_loop,
        )

        result = future.result()

        if stream and isinstance(result, AsyncIterator):
            # If streaming, the result is an async generator. We need to wrap it
            # in a synchronous generator that pulls from it.
            return self._sync_iterator_bridge(result)
        else:
            assert not isinstance(result, AsyncIterator)
            # Could be ClientResponse or list[ClientResponse]
            return result

    def responses(
        self,
        model_id: str,
        **kwargs: Any,
    ) -> ResponseObject | Iterator[ResponseEvent]:
        """Synchronous wrapper for Responses API IPC calls.

        Args:
            model_id: Model ID to execute against.
            kwargs: Same keyword arguments accepted by `aresponses`.
        Returns:
            `ResponseObject` for non-streaming calls, or `Iterator[ResponseEvent]`.
        Example:
            `resp = client.responses("llama3", input="hi")`
            `for event in client.responses("llama3", input="hi", stream=True): ...`
        """
        if (
            not self._sync_loop
            or not self._sync_thread
            or not self._sync_thread.is_alive()
        ):
            self._start_sync_event_loop()

        assert self._sync_loop, "Sync loop not initialized"
        future = asyncio.run_coroutine_threadsafe(
            self.aresponses(model_id, **kwargs),
            self._sync_loop,
        )
        result = future.result()

        if isinstance(result, AsyncIterator):
            return self._sync_iterator_bridge(result)

        assert not isinstance(result, AsyncIterator)
        return result

    async def aresponses_text(
        self,
        model_id: str,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream only text delta chunks from `response.output_text.delta` events."""
        stream = await self.aresponses(model_id, stream=True, **kwargs)
        async for event in stream:
            if isinstance(event, OutputTextDeltaEvent):
                yield event.delta

    def responses_text(
        self,
        model_id: str,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream only text delta chunks from synchronous Responses calls."""
        stream = self.responses(model_id, stream=True, **kwargs)
        for event in stream:
            if isinstance(event, OutputTextDeltaEvent):
                yield event.delta

    def _start_sync_event_loop(self) -> None:
        """Starts a dedicated event loop in a background thread for sync calls."""
        with self._sync_start_lock:
            if self._sync_thread and self._sync_thread.is_alive():
                return

            loop_started = threading.Event()

            def _loop_target():
                self._sync_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._sync_loop)
                loop_started.set()
                self._sync_loop.run_forever()

            self._sync_thread = threading.Thread(
                target=_loop_target, name="pie-client-sync-bridge", daemon=True
            )
            self._sync_thread.start()
            loop_started.wait()

    def _sync_iterator_bridge(self, async_iterator: AsyncIterator[T]) -> Iterator[T]:
        """Bridges an async iterator to a sync iterator."""
        DELTA_TIMEOUT_S = 300  # 5 minutes max wait per delta

        async def _next_delta() -> T:
            return await async_iterator.__anext__()

        try:
            while True:
                if self._ipc_state.engine_dead:
                    raise RuntimeError(
                        "Engine process is dead; cannot receive further deltas."
                    )
                future = asyncio.run_coroutine_threadsafe(
                    _next_delta(),
                    self._sync_loop or asyncio.new_event_loop(),
                )
                try:
                    yield future.result(timeout=DELTA_TIMEOUT_S)
                except TimeoutError as exc:
                    raise RuntimeError(
                        f"Timed out waiting for inference delta after {DELTA_TIMEOUT_S}s. "
                        "Engine may have crashed."
                    ) from exc
                except StopAsyncIteration:
                    break
        finally:
            aclose = getattr(async_iterator, "aclose", None)
            if aclose and self._sync_loop and self._sync_loop.is_running():
                future = asyncio.run_coroutine_threadsafe(aclose(), self._sync_loop)
                with contextlib.suppress(Exception):
                    future.result(timeout=5)

    def close(self):
        """Stops the background event loop thread if it was started."""
        if self._sync_loop and self._sync_loop.is_running():
            self._sync_loop.call_soon_threadsafe(self._sync_loop.stop)
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)

    @staticmethod
    def _normalize_stop_sequences(raw: Any) -> list[str]:
        if not raw:
            return []
        if isinstance(raw, str):
            return [raw]
        result: list[str] = []
        for candidate in raw:
            if not candidate:
                continue
            if candidate not in result:
                result.append(str(candidate))
        return result

    @staticmethod
    def _serialize_optional_payload(value: Any) -> str:
        if not value:
            return ""
        payload = value
        if hasattr(value, "model_dump"):
            payload = value.model_dump()
        elif hasattr(value, "to_dict"):
            payload = value.to_dict()
        return json.dumps(payload)

    @staticmethod
    def _serialize_tools(tools: Any) -> str:
        if not tools:
            return ""
        serializable: list[dict[str, Any]] = []
        for tool in tools:
            if hasattr(tool, "model_dump"):
                serializable.append(tool.model_dump())
            elif hasattr(tool, "to_dict"):
                serializable.append(tool.to_dict())
            elif isinstance(tool, dict):
                serializable.append(tool)
            else:
                raise TypeError(
                    "Tools entries must be dict-like or expose a to_dict() method."
                )
        serializable.sort(key=Client._tool_schema_name)
        return json.dumps(serializable)

    @staticmethod
    def _tool_schema_name(tool: Any) -> str:
        if isinstance(tool, dict):
            function = tool.get("function")
            if isinstance(function, dict):
                return str(function.get("name") or "")
            return str(tool.get("name") or "")
        return ""

    @staticmethod
    def _normalize_tool_choice_payload(value: Any) -> str | dict[str, Any]:
        if value is None:
            return "auto"
        if hasattr(value, "model_dump"):
            return value.model_dump(exclude_none=True)
        if hasattr(value, "to_dict"):
            return value.to_dict()
        if isinstance(value, dict | str):
            return value
        return str(value)

    @staticmethod
    def _extract_usage(deltas: list[ClientDelta]) -> UsageStats:
        usage = UsageStats()
        for delta in deltas:
            if delta.prompt_token_count is not None:
                usage.prompt_tokens = max(usage.prompt_tokens, delta.prompt_token_count)
            if delta.reasoning_tokens is not None:
                usage.reasoning_tokens = max(
                    usage.reasoning_tokens, delta.reasoning_tokens
                )
            if delta.generation_len is not None:
                visible_tokens = max(delta.generation_len - usage.reasoning_tokens, 0)
                usage.completion_tokens = max(usage.completion_tokens, visible_tokens)
        usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
        return usage

    @staticmethod
    def _aggregate_message_text(deltas: list[ClientDelta]) -> str:
        saw_state_events = any(delta.state_events for delta in deltas)
        if not saw_state_events:
            return "".join(filter(None, (delta.content or "" for delta in deltas)))

        parts: list[str] = []
        completed_value: str | None = None
        for delta in deltas:
            for event in delta.state_events:
                if (
                    event.get("item_type") == "message"
                    and event.get("event_type") == "content_delta"
                ):
                    parts.append(str(event.get("delta", "")))
                elif (
                    event.get("item_type") == "message"
                    and event.get("event_type") == "item_completed"
                    and "value" in event
                ):
                    completed_value = str(event["value"])
        if parts:
            return "".join(parts)
        if completed_value is not None:
            return completed_value
        return "".join(parts)

    @staticmethod
    def _aggregate_structured_items(
        deltas: list[ClientDelta],
    ) -> tuple[list[str], list[dict]]:
        reasoning_by_identifier: dict[str, str] = {}
        tool_calls_by_index: dict[int, dict] = {}

        for delta in deltas:
            for event in delta.state_events:
                event_type = event.get("event_type")
                item_type = event.get("item_type")
                identifier = str(event.get("identifier") or item_type or "")

                if item_type == "reasoning":
                    current = reasoning_by_identifier.get(identifier, "")
                    if event_type == "content_delta":
                        reasoning_by_identifier[identifier] = current + str(
                            event.get("delta", "")
                        )
                    elif event_type == "item_completed" and "value" in event:
                        reasoning_by_identifier[identifier] = str(event["value"])
                    continue

                if item_type != "tool_call":
                    continue

                output_index = int(event.get("output_index") or 0)
                tool_call = tool_calls_by_index.setdefault(
                    output_index,
                    {"name": identifier.removeprefix("tool_call:"), "arguments": ""},
                )
                if identifier.startswith("tool_call:"):
                    tool_call["name"] = identifier.removeprefix("tool_call:")

                if identifier == "arguments" and event_type == "content_delta":
                    tool_call["arguments"] = str(tool_call.get("arguments", "")) + str(
                        event.get("delta", "")
                    )
                elif event_type == "item_completed" and "value" in event:
                    value = event["value"]
                    if isinstance(value, str):
                        try:
                            parsed_value = json.loads(value)
                            if isinstance(parsed_value, dict):
                                value = parsed_value
                        except json.JSONDecodeError:
                            pass
                    if isinstance(value, dict):
                        name = value.get("name")
                        arguments = value.get("arguments")
                        if name is not None:
                            tool_call["name"] = str(name)
                        if arguments is not None:
                            tool_call["arguments"] = arguments
                    elif identifier == "arguments":
                        tool_call["arguments"] = value

        reasoning = [
            value for _, value in sorted(reasoning_by_identifier.items()) if value
        ]
        tool_calls = [
            value
            for _, value in sorted(tool_calls_by_index.items())
            if value.get("name")
        ]
        return reasoning, tool_calls

    def _aggregate_response(self, deltas: list[ClientDelta]) -> ClientResponse:
        error_message = next(
            (
                delta.error_message or delta.content or "Inference request failed."
                for delta in deltas
                if delta.error_message or (delta.finish_reason or "").lower() == "error"
            ),
            None,
        )
        if error_message is not None:
            raise InferenceError(error_message)

        aggregated_text = self._aggregate_message_text(deltas)
        finish_reason = next(
            (delta.finish_reason for delta in reversed(deltas) if delta.finish_reason),
            None,
        )
        usage = self._extract_usage(deltas)
        reasoning, tool_calls = self._aggregate_structured_items(deltas)
        return ClientResponse(
            text=aggregated_text,
            finish_reason=finish_reason,
            usage=usage,
            reasoning=reasoning,
            tool_calls=tool_calls,
            deltas=deltas,
        )

    def _aggregate_batch_response(
        self, deltas: list[ClientDelta], batch_size: int
    ) -> list[ClientResponse]:
        """Aggregate deltas into a list of ClientResponses, one per prompt in batch."""
        # Group deltas by prompt_index
        deltas_by_prompt: dict[int, list[ClientDelta]] = {
            i: [] for i in range(batch_size)
        }
        for delta in deltas:
            idx = delta.prompt_index if delta.prompt_index is not None else 0
            if idx < batch_size:
                deltas_by_prompt[idx].append(delta)

        # Create a ClientResponse for each prompt
        return [
            self._aggregate_response(deltas_by_prompt[i]) for i in range(batch_size)
        ]

    def _prepare_prompt_payload(
        self,
        *,
        model_id: str,
        model_path: str,
        formatter: ChatFormatter,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        requested_reasoning_effort = kwargs.get("reasoning_effort")
        native_reasoning = formatter.supports_native_thinking()
        max_generated_tokens = int(
            kwargs.get("max_generated_tokens", MAX_GENERATED_TOKENS)
        )
        requested_reasoning = kwargs.get("reasoning")
        default_reasoning = (
            native_reasoning
            and requested_reasoning is None
            and requested_reasoning_effort is None
        )
        reasoning_flag = bool(
            (requested_reasoning is not False)
            and (requested_reasoning or requested_reasoning_effort or default_reasoning)
            and native_reasoning
        )
        reasoning_effort = (
            requested_reasoning_effort or DEFAULT_BOOLEAN_REASONING_EFFORT
            if reasoning_flag
            else None
        )
        thinking_tokens = (
            formatter.get_thinking_tokens()
            if formatter.supports_native_thinking()
            else {"start": "", "end": ""}
        )
        try:
            (
                messages_for_template,
                image_buffers,
                audio_buffers,
                capabilities,
                content_order,
            ) = build_multimodal_messages(
                formatter=formatter,
                items=messages,
                instructions=kwargs.get("instructions"),
            )
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid chat message payload: {exc}") from exc

        if not messages_for_template:
            raise ValueError("Chat request must include at least one content segment.")

        core_tools_payload = kwargs.get("core_tools")
        if core_tools_payload is None:
            core_tools_payload = kwargs.get("tools")
        active_tools_payload = kwargs.get("active_tools")
        if not active_tools_payload:
            active_tools_payload = core_tools_payload
        if core_tools_payload:
            core_tools_payload = sorted(core_tools_payload, key=self._tool_schema_name)
        if active_tools_payload:
            active_tools_payload = sorted(
                active_tools_payload, key=self._tool_schema_name
            )
        prompt_text = formatter.apply_template(
            messages_for_template,
            reasoning=reasoning_flag,
            reasoning_effort=reasoning_effort,
            task=kwargs.get("task_name"),
            tools=core_tools_payload,
        )
        try:
            layout_segments = build_multimodal_layout(
                prompt_text,
                image_buffers,
                audio_buffers,
                capabilities,
                content_order,
                formatter.image_placeholder,
                formatter.should_clip_image_placeholder,
                audio_placeholder=formatter.get_audio_placeholder(),
                coord_placeholder=formatter.get_coord_placeholder(),
            )
        except ValueError as exc:
            raise ValueError(f"Invalid multimodal layout: {exc}") from exc

        prompt_text = formatter.strip_template_placeholders(prompt_text)

        prompt_bytes = prompt_text.encode("utf-8")
        deterministic = bool(kwargs.get("deterministic", False))
        generation_defaults = formatter.get_generation_defaults(
            "recommended" if deterministic else "default"
        )
        temperature = float(
            self._generation_value(kwargs, generation_defaults, "temperature", 1.0)
        )
        top_p = float(self._generation_value(kwargs, generation_defaults, "top_p", 1.0))
        top_k = int(self._generation_value(kwargs, generation_defaults, "top_k", -1))
        min_p = float(self._generation_value(kwargs, generation_defaults, "min_p", 0.0))
        rng_seed = int(
            kwargs.get("rng_seed", 11 if deterministic else random.randint(0, 2**32 - 1))
        )
        top_logprobs = int(kwargs.get("top_logprobs", 0))
        frequency_penalty = float(
            self._generation_value(
                kwargs, generation_defaults, "frequency_penalty", 0.0
            )
        )
        presence_penalty = float(
            self._generation_value(kwargs, generation_defaults, "presence_penalty", 0.0)
        )
        repetition_context_size = int(
            self._generation_value(
                kwargs, generation_defaults, "repetition_context_size", 60
            )
        )
        repetition_penalty = float(
            self._generation_value(
                kwargs, generation_defaults, "repetition_penalty", 1.0
            )
        )
        logit_bias = {
            int(k): float(v) for k, v in (kwargs.get("logit_bias") or {}).items()
        }
        stop_sequences = self._normalize_stop_sequences(kwargs.get("stop"))
        if not stop_sequences and formatter.control_tokens.end_of_sequence:
            stop_sequences = [formatter.control_tokens.end_of_sequence]
        tool_schemas_json = self._serialize_tools(core_tools_payload)
        active_tool_schemas_json = self._serialize_tools(active_tools_payload)
        response_format_json = self._serialize_optional_payload(
            kwargs.get("response_format")
        )
        num_candidates = max(1, int(kwargs.get("n", 1)))
        best_of = int(kwargs.get("best_of", num_candidates))
        if best_of <= 0:
            best_of = num_candidates
        final_candidates = int(kwargs.get("final_candidates", best_of))
        if final_candidates <= 0:
            final_candidates = best_of
        min_tool_calls = int(kwargs.get("min_tool_calls") or 1)
        max_tool_calls = int(kwargs.get("max_tool_calls") or 0)
        normalized_tool_choice = self._normalize_tool_choice_payload(
            kwargs.get("tool_choice")
        )
        capabilities_payload = [
            {"name": cap.name, "payload": cap.payload, "position": 0}
            for cap in capabilities
        ]
        prompt_payload = {
            "prompt_bytes": prompt_bytes,
            "image_buffers": image_buffers,
            "audio_buffers": audio_buffers,
            "capabilities": capabilities_payload,
            "layout": layout_segments,
            "sampling_params": {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "rng_seed": rng_seed,
                "deterministic": deterministic,
            },
            "logits_params": {
                "top_logprobs": top_logprobs,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "repetition_context_size": repetition_context_size,
                "repetition_penalty": repetition_penalty,
                "logit_bias": logit_bias,
            },
            "max_generated_tokens": max_generated_tokens,
            "stop_sequences": stop_sequences,
            "tool_schemas_json": tool_schemas_json,
            "active_tool_schemas_json": active_tool_schemas_json,
            "response_format_json": response_format_json,
            "num_candidates": num_candidates,
            "best_of": best_of,
            "final_candidates": final_candidates,
            "task_name": kwargs.get("task_name"),
            "reasoning_effort": reasoning_effort,
            "min_tool_calls": min_tool_calls,
            "max_tool_calls": max_tool_calls,
            "tool_calling_tokens": formatter.get_tool_calling_tokens(),
            "output_frame_tokens": formatter.get_output_frame_tokens(),
            "thinking_tokens": thinking_tokens,
            "tool_choice": normalized_tool_choice,
            "prefix_cache": bool(kwargs.get("prefix_cache", True)),
        }
        capture_payload = {
            "model_id": model_id,
            "model_path": model_path,
            "rendered_prompt_text": prompt_text,
            "sampling_params": {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "rng_seed": rng_seed,
                "deterministic": deterministic,
                "top_logprobs": top_logprobs,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "repetition_context_size": repetition_context_size,
                "repetition_penalty": repetition_penalty,
                "logit_bias": logit_bias,
                "n": num_candidates,
                "best_of": best_of,
                "final_candidates": final_candidates,
            },
            "max_generated_tokens": max_generated_tokens,
            "stop_sequences": stop_sequences,
            "tool_schemas_json": tool_schemas_json,
            "response_format_json": response_format_json,
            "task_name": kwargs.get("task_name"),
            "reasoning_effort": reasoning_effort,
            "min_tool_calls": min_tool_calls,
            "max_tool_calls": max_tool_calls,
            "tool_calling_tokens": formatter.get_tool_calling_tokens(),
            "output_frame_tokens": formatter.get_output_frame_tokens(),
            "thinking_tokens": thinking_tokens,
            "tool_choice": normalized_tool_choice,
        }
        return prompt_payload, capture_payload

    @staticmethod
    def _generation_value(
        kwargs: dict[str, Any],
        defaults: dict[str, Any],
        key: str,
        fallback: Any,
    ) -> Any:
        value = kwargs.get(key)
        if value is not None:
            return value
        default_value = defaults.get(key)
        return fallback if default_value is None else default_value

    async def _asubmit_request(
        self, request_id: int, model_id: str, messages: list[dict], **kwargs: Any
    ):
        """Prepares and submits the request over the pynng IPC channel."""
        info = await self._model_registry.get_info(model_id)
        engine_model_id = info.model_id
        prompt_payload, _ = self._prepare_prompt_payload(
            model_id=engine_model_id,
            model_path=info.model_path,
            formatter=info.formatter,
            messages=messages,
            **kwargs,
        )
        response_channel_id = self._ipc_state.response_channel_id or request_id
        logger.debug(
            f"Submitting request {request_id} for model {engine_model_id} with response channel id: {response_channel_id}"
        )
        request_bytes = _build_request_payload(
            request_id=request_id,
            model_id=engine_model_id,
            model_path=info.model_path,
            request_type="generation",
            response_channel_id=response_channel_id,
            prompts=[prompt_payload],
        )
        socket = self._ipc_state.request_socket
        if socket is None:
            raise RuntimeError("Request socket is not initialized.")

        await socket.asend(request_bytes)

    async def _asubmit_request_batch(
        self,
        request_id: int,
        model_id: str,
        conversations: list[list[dict]],
        **kwargs: Any,
    ):
        """Prepares and submits a batched request over the pynng IPC channel."""
        info = await self._model_registry.get_info(model_id)
        engine_model_id = info.model_id

        # Build a prompt payload for each conversation
        prompt_payloads = []
        batch_size = len(conversations)
        for prompt_index, messages in enumerate(conversations):
            prompt_kwargs = self._kwargs_for_prompt(kwargs, prompt_index, batch_size)
            prompt_payload, _ = self._prepare_prompt_payload(
                model_id=engine_model_id,
                model_path=info.model_path,
                formatter=info.formatter,
                messages=messages,
                **prompt_kwargs,
            )
            prompt_payloads.append(prompt_payload)

        response_channel_id = self._ipc_state.response_channel_id or request_id
        logger.debug(
            f"Submitting batched request {request_id} for model {engine_model_id} "
            f"with {len(prompt_payloads)} prompts, response channel: {response_channel_id}"
        )
        request_bytes = _build_request_payload(
            request_id=request_id,
            model_id=engine_model_id,
            model_path=info.model_path,
            request_type="generation",
            response_channel_id=response_channel_id,
            prompts=prompt_payloads,
        )
        socket = self._ipc_state.request_socket
        if socket is None:
            raise RuntimeError("Request socket is not initialized.")

        await socket.asend(request_bytes)
