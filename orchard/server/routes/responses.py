import asyncio
import builtins
import json
import logging
import random
from collections.abc import AsyncIterable
from contextlib import AsyncExitStack
from typing import Any

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from orchard.formatter.multimodal import (
    build_multimodal_layout,
    build_multimodal_messages,
)
from orchard.ipc.serialization import _build_request_payload
from orchard.ipc.utils import (
    ResponseDeltaDict,
    release_delta_resources,
)
from orchard.server.dependencies import IPCStateDep, ModelRegistryDep
from orchard.server.exceptions import InferenceError
from orchard.server.models.reasoning import normalize_reasoning_value
from orchard.server.models.responses import (
    ContentPartAddedEvent,
    ContentPartDoneEvent,
    FunctionCallArgumentsDeltaEvent,
    FunctionCallArgumentsDoneEvent,
    IncompleteDetails,
    InputTokensDetails,
    OutputFunctionCall,
    OutputItemAddedEvent,
    OutputItemDoneEvent,
    OutputMessage,
    OutputReasoning,
    OutputStatus,
    OutputTextContent,
    OutputTextDeltaEvent,
    OutputTextDoneEvent,
    OutputTokensDetails,
    ReasoningContent,
    ReasoningDeltaEvent,
    ReasoningDoneEvent,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
    ResponseInProgressEvent,
    ResponseObject,
    ResponseRequest,
    ResponseStreamState,
    ResponseUsage,
    StreamErrorEvent,
    generate_response_id,
    get_current_timestamp,
)
from orchard.server.routes._common import (
    _ModelNotReadyError,
    extract_usage,
    managed_stream_session,
    resolve_model,
)

logger = logging.getLogger(__name__)

responses_router = APIRouter()


@responses_router.post(
    "/responses",
    response_model=None,  # Disable automatic response model to support streaming
    summary="Create a model response",
    tags=["Responses"],
)
async def handle_response_request(
    request: ResponseRequest,
    ipc_state: IPCStateDep,
    model_registry: ModelRegistryDep,
) -> ResponseObject | EventSourceResponse | JSONResponse:
    """Handle multimodal requests to the `/v1/responses` endpoint."""
    logger.info("Handling response request for model: %s", request.model)

    try:
        canonical_id, model_info = await resolve_model(model_registry, request.model)
    except _ModelNotReadyError as exc:
        return exc.response

    formatter = model_info.formatter

    try:
        messages_for_template, image_buffers, capabilities, content_order = (
            build_multimodal_messages(
                formatter=formatter,
                items=request.get_message_items(),
                instructions=request.instructions,
            )
        )
    except (ValueError, TypeError) as exc:
        logger.error("Invalid multimodal payload for request: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    if not messages_for_template:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Response request must include at least one content segment.",
        )

    try:
        prompt_text = formatter.apply_template(
            messages_for_template,
            reasoning=request.reasoning is not None,
        )
        logger.debug("Prompt text: %s", prompt_text)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to render chat template: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to render chat template.",
        ) from exc

    # Tool definitions — used for both toolbox rendering and PSE grammar compilation
    tools_payload = (
        [tool.to_dict() for tool in request.tools] if request.tools else None
    )
    tool_schemas_json = json.dumps(tools_payload) if tools_payload else ""
    toolbox_text = formatter.render_toolbox(tools_payload) if tools_payload else None

    try:
        layout_segments = build_multimodal_layout(
            prompt_text,
            image_buffers,
            capabilities,
            content_order,
            formatter.control_tokens.start_image_token
            or formatter.default_image_placeholder,
            formatter.should_clip_image_placeholder,
        )
    except ValueError as exc:
        logger.error("Failed to build multimodal layout: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    if formatter.should_clip_image_placeholder:
        prompt_text = prompt_text.replace(formatter.default_image_placeholder, "")

    # Prepend toolbox segment to layout and prompt bytes
    if toolbox_text:
        toolbox_bytes = toolbox_text.encode("utf-8")
        layout_segments.insert(0, {"type": "toolbox", "length": len(toolbox_bytes)})
        prompt_text = toolbox_text + prompt_text

    current_request_id = await ipc_state.get_next_request_id()
    logger.debug(
        "Generated request ID %d for responses submission.", current_request_id
    )
    response_channel_id = ipc_state.response_channel_id or current_request_id

    temperature = request.temperature if request.temperature is not None else 1.0
    top_p = request.top_p if request.top_p is not None else 1.0
    top_k = request.top_k if request.top_k is not None else -1
    min_p = request.min_p if request.min_p is not None else 0.0
    max_output_tokens = request.max_output_tokens or 0
    rng_seed = random.randint(0, 2**32 - 1)

    response_format_json = json.dumps(request.text.to_dict()) if request.text else ""
    reasoning_effort = normalize_reasoning_value(request.reasoning)

    response_queue: asyncio.Queue[ResponseDeltaDict] = asyncio.Queue()
    exit_stack = AsyncExitStack()
    await exit_stack.enter_async_context(
        managed_stream_session(ipc_state, current_request_id, response_queue)
    )

    try:
        prompt_bytes = prompt_text.encode("utf-8")
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
                "top_logprobs": 0,
                "frequency_penalty": request.frequency_penalty or 0.0,
                "logit_bias": {},
                "presence_penalty": request.presence_penalty or 0.0,
                "repetition_context_size": 60,
                "repetition_penalty": 1.0,
            },
            "max_generated_tokens": max_output_tokens,
            "stop_sequences": [],
            "tool_schemas_json": tool_schemas_json,
            "response_format_json": response_format_json,
            "num_candidates": 1,
            "best_of": 1,
            "final_candidates": 1,
            "task": request.task,
            "reasoning_effort": reasoning_effort,
            "max_tool_calls": request.max_tool_calls,
            "tool_calling_tokens": formatter.get_tool_calling_tokens(),
            "tool_choice": request.tool_choice.to_dict() if request.tool_choice else "auto",
        }
        request_bytes = _build_request_payload(
            request_id=current_request_id,
            model_id=canonical_id,
            model_path=model_info.model_path,
            request_type="generation",
            response_channel_id=response_channel_id,
            prompts=[prompt_payload],
        )

        socket = ipc_state.request_socket
        if socket is None:
            raise RuntimeError("Request socket is not initialized.")

        await socket.asend(request_bytes)
        logger.info(
            "Submitted responses request %d with %d layout segments (%d images).",
            current_request_id,
            len(layout_segments),
            len(image_buffers),
        )

        if request.stream:
            logger.debug(
                "Starting streaming response for request %d", current_request_id
            )
            response_id = generate_response_id()

            async def event_stream() -> AsyncIterable[dict[str, str]]:
                try:
                    async for event in stream_response_generator(
                        request_id=current_request_id,
                        queue=response_queue,
                        response_id=response_id,
                        model_name=request.model,
                    ):
                        yield event
                finally:
                    await exit_stack.aclose()

            return EventSourceResponse(
                content=event_stream(),
                media_type="text/event-stream",
            )

        # Non-streaming path
        aggregated = await gather_non_streaming_response(
            current_request_id, response_queue
        )

        usage = ResponseUsage(
            input_tokens=aggregated["prompt_tokens"],
            output_tokens=aggregated["completion_tokens"],
            total_tokens=aggregated["total_tokens"],
            input_tokens_details=InputTokensDetails(
                cached_tokens=aggregated["cached_tokens"],
            ),
            output_tokens_details=OutputTokensDetails(
                reasoning_tokens=aggregated["reasoning_tokens"],
            ),
        )

        incomplete_details = aggregated.get("incomplete_details")
        response = ResponseObject(
            model=request.model,
            output=aggregated["output"],
            usage=usage,
            status=OutputStatus.INCOMPLETE if incomplete_details else OutputStatus.COMPLETED,
            completed_at=aggregated.get("completed_at"),
            incomplete_details=incomplete_details,
            metadata=request.metadata,
            min_p=request.min_p,
            top_p=request.top_p,
            top_k=request.top_k,
            temperature=request.temperature,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            truncation=request.truncation,
            parallel_tool_calls=request.parallel_tool_calls or False,
            tool_choice=request.tool_choice,
            tools=request.tools or [],
            max_tool_calls=request.max_tool_calls,
            text=request.text,
        )

        logger.info("Response request %d completed successfully.", current_request_id)
        await exit_stack.aclose()
        return response
    except HTTPException:
        await exit_stack.aclose()
        raise
    except InferenceError as exc:
        await exit_stack.aclose()
        logger.error(
            "Inference error during response request %d: %s",
            current_request_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive
        await exit_stack.aclose()
        logger.exception(
            "Failed to process multimodal response request %d: %s",
            current_request_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during response generation.",
        ) from exc


async def gather_non_streaming_response(
    request_id: int,
    queue: asyncio.Queue[ResponseDeltaDict],
) -> dict[str, Any]:
    """Aggregate non-streaming deltas into a final response with proper output items."""

    # Track output items by output_index
    output_items: dict[int, dict[str, Any]] = {}
    usage_counts = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
    }
    error_detail: str | None = None
    finish_reason: str | None = None
    completed_at: int | None = None

    while True:
        try:
            delta = await asyncio.wait_for(queue.get(), timeout=30.0)
        except builtins.TimeoutError as exc:
            logger.error(
                "Timeout waiting for response delta for request %d", request_id
            )
            raise InferenceError("Timeout receiving response from engine.") from exc
        else:
            try:
                if not delta:
                    logger.debug(
                        "Received empty delta for response request %d", request_id
                    )
                    continue

                if delta.get("error"):
                    error_detail = str(delta["error"])
                else:
                    status_value = str(delta.get("status", "")).lower()
                    finish_value = str(delta.get("finish_reason", "")).lower()
                    if status_value == "error" or finish_value == "error":
                        error_detail = str(
                            delta.get("content")
                            or delta.get("message")
                            or delta.get("error")
                            or "Response generation failed."
                        )

                # Process state_events from PIE
                state_events = delta.get("state_events", [])
                for event in state_events:
                    _process_state_event_for_output(event, output_items)

                # Also handle legacy content field for backwards compatibility
                if not state_events:
                    delta_content = delta.get("content")
                    if delta_content:
                        if 0 not in output_items:
                            output_items[0] = {
                                "type": "message",
                                "content": "",
                            }
                        output_items[0]["content"] += str(delta_content)

                # Extract usage from delta
                extract_usage(delta, usage_counts)

                # Track finish reason for incomplete_details
                if fr := delta.get("finish_reason"):
                    finish_reason = str(fr).lower()

                if delta.get("is_final_delta", False):
                    completed_at = get_current_timestamp()
                    logger.debug(
                        "Received final delta for response request %d",
                        request_id,
                    )
                    break
            finally:
                queue.task_done()
                if delta:
                    release_delta_resources(delta)

    if error_detail:
        logger.error(
            "Error reported in response stream for request %d: %s",
            request_id,
            error_detail,
        )
        raise InferenceError(error_detail)

    if usage_counts["total_tokens"] <= 0:
        usage_counts["total_tokens"] = (
            usage_counts["prompt_tokens"] + usage_counts["completion_tokens"]
        )

    # Build final output items
    output = _build_output_items(output_items)

    # Determine if response was incomplete
    incomplete_details = None
    if finish_reason in ("length", "max_tokens", "max_output_tokens"):
        incomplete_details = IncompleteDetails(reason="max_output_tokens")
    elif finish_reason == "content_filter":
        incomplete_details = IncompleteDetails(reason="content_filter")

    return {
        "output": output,
        "prompt_tokens": usage_counts["prompt_tokens"],
        "completion_tokens": usage_counts["completion_tokens"],
        "total_tokens": usage_counts["total_tokens"],
        "cached_tokens": usage_counts["cached_tokens"],
        "reasoning_tokens": usage_counts["reasoning_tokens"],
        "completed_at": completed_at,
        "incomplete_details": incomplete_details,
    }


def _process_state_event_for_output(
    event: dict[str, Any],
    output_items: dict[int, dict[str, Any]],
) -> None:
    """Process a single state event and update output_items tracking."""
    event_type = event.get("event_type")
    item_type = event.get("item_type", "message")
    output_index = event.get("output_index", 0)
    identifier = event.get("identifier", "")

    if output_index not in output_items:
        output_items[output_index] = {
            "type": item_type,
            "content": "",
            "identifier": identifier,
        }

    item = output_items[output_index]

    if event_type == "content_delta":
        delta_text = event.get("delta", "")
        item["content"] += delta_text
    elif event_type == "item_completed":
        # Mark as completed, capture final value if present
        item["completed"] = True
        if "value" in event:
            item["value"] = event["value"]


def _build_output_items(
    output_items: dict[int, dict[str, Any]],
) -> list[OutputMessage | OutputFunctionCall | OutputReasoning]:
    """Build final output item models from accumulated state."""
    result: list[OutputMessage | OutputFunctionCall | OutputReasoning] = []

    for output_index in sorted(output_items.keys()):
        item = output_items[output_index]
        item_type = item.get("type", "message")
        content = item.get("content", "")
        identifier = item.get("identifier", "")

        if item_type == "message":
            result.append(
                OutputMessage(
                    status=OutputStatus.COMPLETED,
                    content=[OutputTextContent(text=content)] if content else [],
                )
            )
        elif item_type == "function_call":
            # Extract function name from identifier (format: "tool_call:function_name")
            function_name = ""
            if identifier.startswith("tool_call:"):
                function_name = identifier[len("tool_call:") :]
            result.append(
                OutputFunctionCall(
                    name=function_name,
                    arguments=content,
                    status=OutputStatus.COMPLETED,
                )
            )
        elif item_type == "reasoning":
            result.append(
                OutputReasoning(
                    status=OutputStatus.COMPLETED,
                    content=[ReasoningContent(text=content)] if content else [],
                )
            )

    # If no items were built, create a default empty message
    if not result:
        result.append(OutputMessage(status=OutputStatus.COMPLETED, content=[]))

    return result


async def stream_response_generator(
    request_id: int,
    queue: asyncio.Queue[ResponseDeltaDict],
    response_id: str,
    model_name: str,
) -> AsyncIterable[dict[str, str]]:
    """Generate Open Responses SSE events from PIE deltas with state_events."""
    stream_state = ResponseStreamState(response_id=response_id, model=model_name)

    # Emit response.created
    yield _format_sse_event(
        ResponseCreatedEvent(
            sequence_number=stream_state.next_sequence_number(),
            response=stream_state.snapshot(),
        )
    )

    # Emit response.in_progress
    yield _format_sse_event(
        ResponseInProgressEvent(
            sequence_number=stream_state.next_sequence_number(),
            response=stream_state.snapshot(),
        )
    )

    error_occurred = False
    error_detail: str | None = None
    finish_reason: str | None = None

    while True:
        try:
            delta = await asyncio.wait_for(queue.get(), timeout=30.0)
        except TimeoutError:
            logger.error(
                "Timeout waiting for delta for streaming request %d", request_id
            )
            error_occurred = True
            error_detail = "Timeout receiving response from engine."
            break
        else:
            try:
                if not delta:
                    continue

                if delta.get("error"):
                    error_occurred = True
                    error_detail = str(delta["error"])
                    break

                status_value = str(delta.get("status", "")).lower()
                finish_value = str(delta.get("finish_reason", "")).lower()
                if status_value == "error" or finish_value == "error":
                    error_occurred = True
                    error_detail = str(
                        delta.get("content")
                        or delta.get("message")
                        or delta.get("error")
                        or "Response generation failed."
                    )
                    break

                # Process state_events
                state_events = delta.get("state_events", [])
                for event in state_events:
                    async for sse_event in _process_state_event_for_streaming(
                        event, stream_state
                    ):
                        yield sse_event

                # Handle legacy content field for backwards compatibility
                if not state_events:
                    delta_content = delta.get("content")
                    if delta_content:
                        # Treat as message content delta
                        item = stream_state.get_or_create_item(0, "message")
                        if item.accumulated_content == "":
                            # First content - emit item added events
                            yield _format_sse_event(
                                OutputItemAddedEvent(
                                    sequence_number=stream_state.next_sequence_number(),
                                    output_index=0,
                                    item=item.to_skeleton(),
                                )
                            )
                            yield _format_sse_event(
                                ContentPartAddedEvent(
                                    sequence_number=stream_state.next_sequence_number(),
                                    item_id=item.item_id,
                                    output_index=0,
                                    content_index=0,
                                    part=OutputTextContent(text=""),
                                )
                            )
                        item.accumulated_content += str(delta_content)
                        yield _format_sse_event(
                            OutputTextDeltaEvent(
                                sequence_number=stream_state.next_sequence_number(),
                                item_id=item.item_id,
                                output_index=0,
                                content_index=0,
                                delta=str(delta_content),
                            )
                        )

                # Extract usage info
                if usage := delta.get("usage"):
                    if isinstance(usage, dict):
                        stream_state.usage = ResponseUsage(
                            input_tokens=usage.get("input_tokens", 0),
                            output_tokens=usage.get("output_tokens", 0),
                            total_tokens=usage.get("total_tokens", 0),
                            input_tokens_details=InputTokensDetails(
                                cached_tokens=usage.get("cached_tokens", 0),
                            ),
                            output_tokens_details=OutputTokensDetails(
                                reasoning_tokens=usage.get("reasoning_tokens", 0),
                            ),
                        )

                # Extract token details from delta (PIE sends these at top level)
                cached_tokens = delta.get("cached_token_count")
                reasoning_tokens = delta.get("reasoning_tokens")
                if cached_tokens is not None or reasoning_tokens is not None:
                    current_usage = stream_state.usage or ResponseUsage(
                        input_tokens=0, output_tokens=0, total_tokens=0
                    )
                    stream_state.usage = ResponseUsage(
                        input_tokens=current_usage.input_tokens,
                        output_tokens=current_usage.output_tokens,
                        total_tokens=current_usage.total_tokens,
                        input_tokens_details=InputTokensDetails(
                            cached_tokens=cached_tokens or 0,
                        ),
                        output_tokens_details=OutputTokensDetails(
                            reasoning_tokens=reasoning_tokens or 0,
                        ),
                    )

                # Track finish reason for incomplete detection
                if fr := delta.get("finish_reason"):
                    finish_reason = str(fr).lower()

                if delta.get("is_final_delta", False):
                    logger.debug(
                        "Received final delta for streaming request %d", request_id
                    )
                    break
            finally:
                queue.task_done()
                if delta:
                    release_delta_resources(delta)

    # Emit completion events for items that didn't receive an item_completed event
    # from PIE. Items that did receive item_completed already have status=COMPLETED
    # (set in _process_state_event_for_streaming) and are skipped here.
    for output_index, item in sorted(stream_state.items.items()):
        if item.status != OutputStatus.COMPLETED:
            item.status = OutputStatus.COMPLETED
            # Emit done events based on item type
            if item.item_type == "message":
                yield _format_sse_event(
                    OutputTextDoneEvent(
                        sequence_number=stream_state.next_sequence_number(),
                        item_id=item.item_id,
                        output_index=output_index,
                        content_index=0,
                        text=item.accumulated_content,
                    )
                )
                yield _format_sse_event(
                    ContentPartDoneEvent(
                        sequence_number=stream_state.next_sequence_number(),
                        item_id=item.item_id,
                        output_index=output_index,
                        content_index=0,
                        part=OutputTextContent(text=item.accumulated_content),
                    )
                )
            elif item.item_type == "function_call":
                yield _format_sse_event(
                    FunctionCallArgumentsDoneEvent(
                        sequence_number=stream_state.next_sequence_number(),
                        item_id=item.item_id,
                        output_index=output_index,
                        arguments=item.accumulated_content,
                    )
                )
            elif item.item_type == "reasoning":
                yield _format_sse_event(
                    ReasoningDoneEvent(
                        sequence_number=stream_state.next_sequence_number(),
                        item_id=item.item_id,
                        output_index=output_index,
                        content_index=0,
                        text=item.accumulated_content,
                    )
                )

            yield _format_sse_event(
                OutputItemDoneEvent(
                    sequence_number=stream_state.next_sequence_number(),
                    output_index=output_index,
                    item=item.to_completed(),
                )
            )

    # Emit final response event
    if error_occurred:
        stream_state.status = OutputStatus.FAILED
        yield _format_sse_event(
            ResponseFailedEvent(
                sequence_number=stream_state.next_sequence_number(),
                response=stream_state.snapshot(),
            )
        )
        if error_detail:
            yield _format_sse_event(
                StreamErrorEvent(
                    sequence_number=stream_state.next_sequence_number(),
                    error={"message": error_detail, "type": "inference_error"},
                )
            )
    else:
        # Set completed_at timestamp
        stream_state.completed_at = get_current_timestamp()

        # Determine if response was incomplete (truncated)
        is_incomplete = finish_reason in ("length", "max_tokens", "max_output_tokens")
        if is_incomplete:
            stream_state.status = OutputStatus.INCOMPLETE
            stream_state.incomplete_details = IncompleteDetails(
                reason="max_output_tokens"
            )
            yield _format_sse_event(
                ResponseIncompleteEvent(
                    sequence_number=stream_state.next_sequence_number(),
                    response=stream_state.snapshot(),
                )
            )
        elif finish_reason == "content_filter":
            stream_state.status = OutputStatus.INCOMPLETE
            stream_state.incomplete_details = IncompleteDetails(reason="content_filter")
            yield _format_sse_event(
                ResponseIncompleteEvent(
                    sequence_number=stream_state.next_sequence_number(),
                    response=stream_state.snapshot(),
                )
            )
        else:
            stream_state.status = OutputStatus.COMPLETED
            yield _format_sse_event(
                ResponseCompletedEvent(
                    sequence_number=stream_state.next_sequence_number(),
                    response=stream_state.snapshot(),
                )
            )

    # Emit terminal [DONE] event (required by spec)
    yield {"data": "[DONE]"}

    logger.info("SSE stream for responses request %d completed", request_id)


async def _process_state_event_for_streaming(
    event: dict[str, Any],
    stream_state: ResponseStreamState,
) -> AsyncIterable[dict[str, str]]:
    """Process a single PSE state event and yield Open Responses SSE events."""
    event_type = event.get("event_type")
    item_type = event.get("item_type", "message")
    output_index = event.get("output_index", 0)
    identifier = event.get("identifier", "")

    item = stream_state.get_or_create_item(output_index, item_type, identifier)

    if event_type == "item_started":
        # Emit output_item.added
        yield _format_sse_event(
            OutputItemAddedEvent(
                sequence_number=stream_state.next_sequence_number(),
                output_index=output_index,
                item=item.to_skeleton(),
            )
        )
        # Emit content_part.added for items with content
        if item_type in ("message", "reasoning"):
            if item_type == "message":
                part = OutputTextContent(text="")
            else:
                part = ReasoningContent(text="")
            yield _format_sse_event(
                ContentPartAddedEvent(
                    sequence_number=stream_state.next_sequence_number(),
                    item_id=item.item_id,
                    output_index=output_index,
                    content_index=0,
                    part=part,
                )
            )

    elif event_type == "content_delta":
        delta_text = event.get("delta", "")
        item.accumulated_content += delta_text

        if item_type == "message":
            yield _format_sse_event(
                OutputTextDeltaEvent(
                    sequence_number=stream_state.next_sequence_number(),
                    item_id=item.item_id,
                    output_index=output_index,
                    content_index=0,
                    delta=delta_text,
                )
            )
        elif item_type == "function_call":
            yield _format_sse_event(
                FunctionCallArgumentsDeltaEvent(
                    sequence_number=stream_state.next_sequence_number(),
                    item_id=item.item_id,
                    output_index=output_index,
                    delta=delta_text,
                )
            )
        elif item_type == "reasoning":
            yield _format_sse_event(
                ReasoningDeltaEvent(
                    sequence_number=stream_state.next_sequence_number(),
                    item_id=item.item_id,
                    output_index=output_index,
                    content_index=0,
                    delta=delta_text,
                )
            )

    elif event_type == "item_completed":
        item.status = OutputStatus.COMPLETED
        # Emit done events
        if item_type == "message":
            yield _format_sse_event(
                OutputTextDoneEvent(
                    sequence_number=stream_state.next_sequence_number(),
                    item_id=item.item_id,
                    output_index=output_index,
                    content_index=0,
                    text=item.accumulated_content,
                )
            )
            yield _format_sse_event(
                ContentPartDoneEvent(
                    sequence_number=stream_state.next_sequence_number(),
                    item_id=item.item_id,
                    output_index=output_index,
                    content_index=0,
                    part=OutputTextContent(text=item.accumulated_content),
                )
            )
        elif item_type == "function_call":
            yield _format_sse_event(
                FunctionCallArgumentsDoneEvent(
                    sequence_number=stream_state.next_sequence_number(),
                    item_id=item.item_id,
                    output_index=output_index,
                    arguments=item.accumulated_content,
                )
            )
        elif item_type == "reasoning":
            yield _format_sse_event(
                ReasoningDoneEvent(
                    sequence_number=stream_state.next_sequence_number(),
                    item_id=item.item_id,
                    output_index=output_index,
                    content_index=0,
                    text=item.accumulated_content,
                )
            )

        yield _format_sse_event(
            OutputItemDoneEvent(
                sequence_number=stream_state.next_sequence_number(),
                output_index=output_index,
                item=item.to_completed(),
            )
        )


def _format_sse_event(event: Any) -> dict[str, str]:
    """Format a streaming event for SSE transport."""
    return {
        "event": event.type,
        "data": event.model_dump_json(exclude_none=True),
    }
