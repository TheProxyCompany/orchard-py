import asyncio
import builtins
import json
import logging
import random
from typing import Any

from fastapi import APIRouter, HTTPException, status

from orchard.app.ipc_dispatch import QueueRegistration
from orchard.app.model_registry import (
    ModelLoadState,
    ModelResolutionError,
)
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
    OutputMessage,
    OutputTextContent,
    ResponseObject,
    ResponseRequest,
    ResponseUsage,
)

logger = logging.getLogger(__name__)

responses_router = APIRouter()


@responses_router.post(
    "/responses",
    response_model=ResponseObject,
    summary="Create a model response",
    tags=["Responses"],
)
async def handle_response_request(
    request: ResponseRequest,
    ipc_state: IPCStateDep,
    model_registry: ModelRegistryDep,
) -> ResponseObject:
    """Handle multimodal requests to the `/v1/responses` endpoint."""
    logger.info("Handling response request for model: %s", request.model)

    try:
        model_state, canonical_id = await model_registry.schedule_model(request.model)
    except ModelResolutionError as exc:
        logger.error("Model resolution failed for %s: %s", request.model, exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": str(exc), "candidates": exc.candidates},
        ) from exc

    if model_state == ModelLoadState.LOADING:
        logger.info("Model %s still loading for responses request.", canonical_id)
        return ResponseObject(
            model=request.model,
            output=[
                OutputMessage(
                    content=[
                        OutputTextContent(
                            text="Model is loading. Please retry once initialization completes."
                        )
                    ]
                )
            ],
            usage=ResponseUsage(input_tokens=0, output_tokens=0, total_tokens=0),
            min_p=request.min_p,
            top_p=request.top_p,
            top_k=request.top_k,
            temperature=request.temperature,
            parallel_tool_calls=request.parallel_tool_calls or False,
            tool_choice=request.tool_choice,
            tools=request.tools or [],
            text=request.text,
        )

    if model_state == ModelLoadState.FAILED:
        error_detail = model_registry.get_error(canonical_id) or "Model failed to load."
        logger.error("Model %s failed to load: %s", canonical_id, error_detail)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=error_detail,
        )

    model_info = model_registry.get_if_ready(canonical_id)
    if not model_info:
        logger.error(
            "Model registry reported READY but runtime info missing for %s",
            canonical_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                f"Model '{canonical_id}' reported READY but no runtime info was found."
            ),
        )

    formatter = model_info.formatter

    try:
        messages_for_template, image_buffers, capabilities, content_order = (
            build_multimodal_messages(
                formatter=formatter,
                items=request.items,
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
            messages_for_template, reasoning=request.reasoning is not None
        )
        logger.debug("Prompt text: %s", prompt_text)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to render chat template: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to render chat template.",
        ) from exc

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

    tools_payload = (
        [tool.to_dict() for tool in request.tools] if request.tools else None
    )
    tool_schemas_json = json.dumps(tools_payload) if tools_payload else ""

    response_format_json = json.dumps(request.text.to_dict()) if request.text else ""
    reasoning_effort = normalize_reasoning_value(request.reasoning)

    response_queue: asyncio.Queue[ResponseDeltaDict] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    ipc_state.active_request_queues[current_request_id] = QueueRegistration(
        loop=loop, queue=response_queue
    )
    logger.debug("Registered queue for response request %d.", current_request_id)

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
                "frequency_penalty": 0.0,
                "logit_bias": {},
                "presence_penalty": 0.0,
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
            "tool_calling_tokens": {
                "call_start": formatter.control_tokens.tool_calling.call_start,
                "call_end": formatter.control_tokens.tool_calling.call_end,
                "section_start": formatter.control_tokens.tool_calling.section_start,
                "section_end": formatter.control_tokens.tool_calling.section_end,
                "name_separator": formatter.control_tokens.tool_calling.name_separator,
            },
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
            logger.info(
                "Streaming requested for response %d but not yet implemented.",
                current_request_id,
            )
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Streaming responses are not yet supported for this endpoint.",
            )

        aggregated = await gather_non_streaming_response(
            current_request_id, response_queue
        )

        usage = ResponseUsage(
            input_tokens=aggregated["prompt_tokens"],
            output_tokens=aggregated["completion_tokens"],
            total_tokens=aggregated["total_tokens"],
        )

        response_message = OutputMessage(
            content=[OutputTextContent(text=aggregated["content"])]
        )

        response = ResponseObject(
            model=request.model,
            output=[response_message],
            usage=usage,
            min_p=request.min_p,
            top_p=request.top_p,
            top_k=request.top_k,
            temperature=request.temperature,
            parallel_tool_calls=request.parallel_tool_calls or False,
            tool_choice=request.tool_choice,
            tools=request.tools or [],
            text=request.text,
        )

        logger.info("Response request %d completed successfully.", current_request_id)
        return response
    except HTTPException:
        raise
    except InferenceError as exc:
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
        logger.exception(
            "Failed to process multimodal response request %d: %s",
            current_request_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during response generation.",
        ) from exc
    finally:
        binding = ipc_state.active_request_queues.pop(current_request_id, None)
        if binding is not None:
            queue = binding.queue
            logger.debug(
                "Cleaning up queue for response request %d.", current_request_id
            )
            try:
                while True:
                    leftover = queue.get_nowait()
                    release_delta_resources(leftover)
                    queue.task_done()
            except asyncio.QueueEmpty:
                pass


async def gather_non_streaming_response(
    request_id: int,
    queue: asyncio.Queue[ResponseDeltaDict],
) -> dict[str, Any]:
    """Aggregate non-streaming deltas into a final multimodal response payload."""

    def _coerce_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            logger.debug(
                "Failed to coerce value %s to int for request %d",
                value,
                request_id,
            )
            return None

    accumulated_parts: list[str] = []
    usage_counts = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    usage_payload: dict[str, Any] | None = None
    error_detail: str | None = None

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
                else:
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

                    delta_content = delta.get("content")
                    if delta_content:
                        accumulated_parts.append(str(delta_content))

                    if usage := delta.get("usage"):
                        if isinstance(usage, dict):
                            usage_payload = usage

                    for source_key, target_key in (
                        ("prompt_token_count", "prompt_tokens"),
                        ("prompt_tokens", "prompt_tokens"),
                        ("input_tokens", "prompt_tokens"),
                        ("completion_token_count", "completion_tokens"),
                        ("completion_tokens", "completion_tokens"),
                        ("output_tokens", "completion_tokens"),
                        ("total_token_count", "total_tokens"),
                        ("total_tokens", "total_tokens"),
                    ):
                        if source_key in delta:
                            value = _coerce_int(delta.get(source_key))
                            if value is not None:
                                usage_counts[target_key] = value

                    if usage_payload:
                        for source_key, target_key in (
                            ("prompt_tokens", "prompt_tokens"),
                            ("input_tokens", "prompt_tokens"),
                            ("completion_tokens", "completion_tokens"),
                            ("output_tokens", "completion_tokens"),
                            ("total_tokens", "total_tokens"),
                        ):
                            if source_key in usage_payload:
                                value = _coerce_int(usage_payload.get(source_key))
                                if value is not None:
                                    usage_counts[target_key] = value

                    if delta.get("is_final_delta", False):
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

    return {
        "content": "".join(accumulated_parts),
        "prompt_tokens": usage_counts["prompt_tokens"],
        "completion_tokens": usage_counts["completion_tokens"],
        "total_tokens": usage_counts["total_tokens"],
    }
