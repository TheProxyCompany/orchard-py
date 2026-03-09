from __future__ import annotations

import asyncio
import json
import logging
import random
import threading
from collections.abc import AsyncIterator, Iterator
from typing import Any, Literal, TypeVar, overload

from orchard.app.ipc_dispatch import IPCState, QueueRegistration
from orchard.app.model_registry import ModelRegistry
from orchard.clients.responses import (
    ResponseEvent,
    ResponsesRequest,
    aggregate_non_streaming_response,
    iter_response_events,
)
from orchard.engine import ClientDelta, ClientResponse, UsageStats
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
from orchard.server.models.responses import OutputTextDeltaEvent, ResponseObject

logger = logging.getLogger(__name__)
T = TypeVar("T")


class Client:
    """
    A high-level Python client for the Proxy Inference Engine (PIE).

    Provides both synchronous and asynchronous interfaces for interacting
    with the shared, on-demand engine service managed by InferenceEngine.
    """

    def __init__(self, ipc_state: IPCState, model_registry: ModelRegistry):
        self._ipc_state = ipc_state
        self._model_registry = model_registry

        # For the synchronous wrapper
        self._sync_loop: asyncio.AbstractEventLoop | None = None
        self._sync_thread: threading.Thread | None = None

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

    async def _async_process_stream(
        self, response_queue: asyncio.Queue[ResponseDeltaDict]
    ) -> AsyncIterator[ClientDelta]:
        """The core async stream processor."""
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
                should_stop = client_delta.is_final

                try:
                    yield client_delta
                finally:
                    release_delta_resources(
                        delta
                    )  # Release resources after consumer is done

                if should_stop:
                    break
        finally:
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
    ) -> AsyncIterator[ResponseDeltaDict]:
        """Yield raw response deltas for Responses API event mapping."""
        # Intentionally mirrors _async_process_stream, but yields raw payloads
        # (including state_events) instead of ClientDelta objects.
        try:
            while True:
                delta = await response_queue.get()
                normalise_delta_payload(delta)
                should_stop = bool(delta.get("is_final_delta"))

                try:
                    yield delta
                finally:
                    release_delta_resources(delta)

                if should_stop:
                    break
        finally:
            while not response_queue.empty():
                try:
                    leftover = response_queue.get_nowait()
                    release_delta_resources(leftover)
                except asyncio.QueueEmpty:
                    break

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
                try:
                    async for delta in self._async_process_stream(response_queue):
                        yield delta
                finally:
                    _cleanup_queue()

            if stream:
                stream_managed_cleanup = True
                return _stream_generator()
            else:
                deltas = [
                    delta async for delta in self._async_process_stream(response_queue)
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
                try:
                    async for event in iter_response_events(
                        self._async_process_raw_stream(response_queue),
                        model_id=model_id,
                    ):
                        yield event
                finally:
                    _cleanup_queue()

            if response_request.stream:
                stream_managed_cleanup = True
                return _response_stream()

            deltas = [
                delta async for delta in self._async_process_raw_stream(response_queue)
            ]
            _cleanup_queue()
            return aggregate_non_streaming_response(
                deltas, model_id, response_request
            )
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
        loop_started.wait()  # Ensure the loop is running before we try to use it

    def _sync_iterator_bridge(self, async_iterator: AsyncIterator[T]) -> Iterator[T]:
        """Bridges an async iterator to a sync iterator."""

        async def _next_delta() -> T:
            return await async_iterator.__anext__()

        while True:
            future = asyncio.run_coroutine_threadsafe(
                _next_delta(),
                self._sync_loop or asyncio.new_event_loop(),
            )
            try:
                yield future.result()
            except StopAsyncIteration:
                break

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
        return json.dumps(serializable)

    @staticmethod
    def _extract_usage(deltas: list[ClientDelta]) -> UsageStats:
        usage = UsageStats()
        for delta in deltas:
            if delta.prompt_token_count is not None:
                usage.prompt_tokens = max(usage.prompt_tokens, delta.prompt_token_count)
            if delta.generation_len is not None:
                usage.completion_tokens = max(
                    usage.completion_tokens, delta.generation_len
                )
        usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
        return usage

    def _aggregate_response(self, deltas: list[ClientDelta]) -> ClientResponse:
        aggregated_text = "".join(
            filter(None, (delta.content or "" for delta in deltas))
        )
        finish_reason = next(
            (delta.finish_reason for delta in reversed(deltas) if delta.finish_reason),
            None,
        )
        usage = self._extract_usage(deltas)
        return ClientResponse(
            text=aggregated_text,
            finish_reason=finish_reason,
            usage=usage,
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

    async def _asubmit_request(
        self, request_id: int, model_id: str, messages: list[dict], **kwargs: Any
    ):
        """Prepares and submits the request over the pynng IPC channel."""
        info = await self._model_registry.get_info(model_id)

        reasoning_effort = kwargs.get("reasoning_effort")
        reasoning_flag = bool(kwargs.get("reasoning") or reasoning_effort)
        try:
            messages_for_template, image_buffers, capabilities, content_order = (
                build_multimodal_messages(
                    formatter=info.formatter,
                    items=messages,
                    instructions=kwargs.get("instructions"),
                )
            )
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid chat message payload: {exc}") from exc

        if not messages_for_template:
            raise ValueError("Chat request must include at least one content segment.")

        tools_payload = kwargs.get("tools")
        prompt_text = info.formatter.apply_template(
            messages_for_template,
            reasoning=reasoning_flag,
            task=kwargs.get("task_name"),
            tools=tools_payload,
        )
        try:
            layout_segments = build_multimodal_layout(
                prompt_text,
                image_buffers,
                capabilities,
                content_order,
                info.formatter.control_tokens.start_image_token
                or info.formatter.default_image_placeholder,
                info.formatter.should_clip_image_placeholder,
                coord_placeholder=info.formatter.get_coord_placeholder(),
            )
        except ValueError as exc:
            raise ValueError(f"Invalid multimodal layout: {exc}") from exc

        if info.formatter.should_clip_image_placeholder:
            prompt_text = prompt_text.replace(
                info.formatter.default_image_placeholder, ""
            )

        # Strip coord placeholders from prompt text (they're handled by layout segments)
        coord_placeholder = info.formatter.get_coord_placeholder()
        if coord_placeholder:
            prompt_text = prompt_text.replace(coord_placeholder, "")

        prompt_bytes = prompt_text.encode("utf-8")

        temperature = float(kwargs.get("temperature", 1.0))
        top_p = float(kwargs.get("top_p", 1.0))
        top_k = int(kwargs.get("top_k", -1))
        min_p = float(kwargs.get("min_p", 0.0))
        rng_seed = int(kwargs.get("rng_seed", random.randint(0, 2**32 - 1)))
        max_generated_tokens = int(kwargs.get("max_generated_tokens", 1024))
        top_logprobs = int(kwargs.get("top_logprobs", 0))
        frequency_penalty = float(kwargs.get("frequency_penalty", 0.0))
        presence_penalty = float(kwargs.get("presence_penalty", 0.0))
        repetition_context_size = int(kwargs.get("repetition_context_size", 60))
        repetition_penalty = float(kwargs.get("repetition_penalty", 1.0))

        logit_bias = {
            int(k): float(v) for k, v in (kwargs.get("logit_bias") or {}).items()
        }

        stop_sequences = self._normalize_stop_sequences(kwargs.get("stop"))
        tool_schemas_json = self._serialize_tools(tools_payload)
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

        response_channel_id = self._ipc_state.response_channel_id or request_id
        # Convert capability inputs to serialization format
        capabilities_payload = [
            {"name": cap.name, "payload": cap.payload, "position": 0}
            for cap in capabilities
        ]

        prompt_payload = {
            "prompt_bytes": prompt_bytes,
            "image_buffers": image_buffers,
            "capabilities": capabilities_payload,
            "layout": layout_segments,
            "sampling_params": {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "rng_seed": rng_seed,
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
            "response_format_json": response_format_json,
            "num_candidates": num_candidates,
            "best_of": best_of,
            "final_candidates": final_candidates,
            "task_name": kwargs.get("task_name"),
            "reasoning_effort": reasoning_effort,
            "tool_calling_tokens": info.formatter.get_tool_calling_tokens(),
            "tool_choice": kwargs.get("tool_choice", "auto"),
        }
        logger.debug(
            f"Submitting request {request_id} for model {model_id} with response channel id: {response_channel_id}"
        )
        request_bytes = _build_request_payload(
            request_id=request_id,
            model_id=model_id,
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

        reasoning_effort = kwargs.get("reasoning_effort")
        reasoning_flag = bool(kwargs.get("reasoning") or reasoning_effort)

        # Shared generation parameters
        temperature = float(kwargs.get("temperature", 1.0))
        top_p = float(kwargs.get("top_p", 1.0))
        top_k = int(kwargs.get("top_k", -1))
        min_p = float(kwargs.get("min_p", 0.0))
        max_generated_tokens = int(kwargs.get("max_generated_tokens", 1024))
        top_logprobs = int(kwargs.get("top_logprobs", 0))
        frequency_penalty = float(kwargs.get("frequency_penalty", 0.0))
        presence_penalty = float(kwargs.get("presence_penalty", 0.0))
        repetition_context_size = int(kwargs.get("repetition_context_size", 60))
        repetition_penalty = float(kwargs.get("repetition_penalty", 1.0))
        logit_bias = {
            int(k): float(v) for k, v in (kwargs.get("logit_bias") or {}).items()
        }
        stop_sequences = self._normalize_stop_sequences(kwargs.get("stop"))
        tools_payload = kwargs.get("tools")
        tool_schemas_json = self._serialize_tools(tools_payload)
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

        # Build a prompt payload for each conversation
        prompt_payloads = []
        for messages in conversations:
            try:
                messages_for_template, image_buffers, capabilities, content_order = (
                    build_multimodal_messages(
                        formatter=info.formatter,
                        items=messages,
                        instructions=kwargs.get("instructions"),
                    )
                )
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid chat message payload: {exc}") from exc

            if not messages_for_template:
                raise ValueError(
                    "Chat request must include at least one content segment."
                )

            prompt_text = info.formatter.apply_template(
                messages_for_template,
                reasoning=reasoning_flag,
                task=kwargs.get("task_name"),
                tools=tools_payload,
            )
            try:
                layout_segments = build_multimodal_layout(
                    prompt_text,
                    image_buffers,
                    capabilities,
                    content_order,
                    info.formatter.control_tokens.start_image_token
                    or info.formatter.default_image_placeholder,
                    info.formatter.should_clip_image_placeholder,
                    coord_placeholder=info.formatter.get_coord_placeholder(),
                )
            except ValueError as exc:
                raise ValueError(f"Invalid multimodal layout: {exc}") from exc

            if info.formatter.should_clip_image_placeholder:
                prompt_text = prompt_text.replace(
                    info.formatter.default_image_placeholder, ""
                )

            coord_placeholder = info.formatter.get_coord_placeholder()
            if coord_placeholder:
                prompt_text = prompt_text.replace(coord_placeholder, "")

            prompt_bytes = prompt_text.encode("utf-8")
            capabilities_payload = [
                {"name": cap.name, "payload": cap.payload, "position": 0}
                for cap in capabilities
            ]

            # Each prompt gets its own rng seed
            rng_seed = int(kwargs.get("rng_seed", random.randint(0, 2**32 - 1)))

            prompt_payloads.append(
                {
                    "prompt_bytes": prompt_bytes,
                    "image_buffers": image_buffers,
                    "capabilities": capabilities_payload,
                    "layout": layout_segments,
                    "sampling_params": {
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                        "min_p": min_p,
                        "rng_seed": rng_seed,
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
                    "response_format_json": response_format_json,
                    "num_candidates": num_candidates,
                    "best_of": best_of,
                    "final_candidates": final_candidates,
                    "task_name": kwargs.get("task_name"),
                    "reasoning_effort": reasoning_effort,
                    "tool_calling_tokens": info.formatter.get_tool_calling_tokens(),
                    "tool_choice": kwargs.get("tool_choice", "auto"),
                }
            )

        response_channel_id = self._ipc_state.response_channel_id or request_id
        logger.debug(
            f"Submitting batched request {request_id} for model {model_id} "
            f"with {len(prompt_payloads)} prompts, response channel: {response_channel_id}"
        )
        request_bytes = _build_request_payload(
            request_id=request_id,
            model_id=model_id,
            model_path=info.model_path,
            request_type="generation",
            response_channel_id=response_channel_id,
            prompts=prompt_payloads,
        )
        socket = self._ipc_state.request_socket
        if socket is None:
            raise RuntimeError("Request socket is not initialized.")

        await socket.asend(request_bytes)
