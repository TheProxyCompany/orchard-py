import asyncio
import logging
import random
from typing import Any

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from orchard.app.ipc_dispatch import QueueRegistration
from orchard.app.model_registry import (
    ModelLoadState,
    ModelResolutionError,
)
from orchard.ipc.serialization import _build_request_payload
from orchard.ipc.utils import (
    ResponseDeltaDict,
    release_delta_resources,
)
from orchard.server.dependencies import IPCStateDep, ModelRegistryDep
from orchard.server.exceptions import InferenceError
from orchard.server.models.chat.logprobs import ChatCompletionLogProbs
from orchard.server.models.chat.output import ChatCompletionChoice
from orchard.server.models.completions import (
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    CompletionUsage,
)
from orchard.server.routes.chat import (
    gather_non_streaming_batch_response,
)

logger = logging.getLogger(__name__)

completions_router = APIRouter()


@completions_router.post(
    "/completions",
    response_model=CompletionResponse,
    summary="Create a text completion",
    tags=["Completions"],
)
async def handle_completion_request(
    request: CompletionRequest,
    ipc_state: IPCStateDep,
    model_registry: ModelRegistryDep,
) -> CompletionResponse | JSONResponse:
    """Handles requests to the `/v1/completions` endpoint."""
    logger.info("Handling completion request for model: %s", request.model)

    if request.stream:
        logger.warning("Streaming requested for /completions but not supported.")
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Streaming is not supported in this version.",
        )

    prompts = _normalize_prompt_inputs(request.prompt)

    try:
        model_state, canonical_id = await model_registry.schedule_model(request.model)
    except ModelResolutionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": str(exc), "candidates": exc.candidates},
        ) from exc

    if model_state in {ModelLoadState.LOADING, ModelLoadState.DOWNLOADING}:
        status_text = (
            "downloading" if model_state == ModelLoadState.DOWNLOADING else "loading"
        )
        payload = {
            "status": status_text,
            "message": "Model download in progress. Retry after a short delay.",
            "model_id": canonical_id,
        }
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content=payload,
            headers={"Retry-After": "30"},
        )

    if model_state == ModelLoadState.FAILED:
        error_detail = model_registry.get_error(canonical_id) or "Model failed to load."
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=error_detail,
        )

    model_info = model_registry.get_if_ready(canonical_id)
    if not model_info:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                f"Model '{canonical_id}' reported READY but no runtime info was found."
            ),
        )

    formatter = model_info.formatter
    model_path = model_info.model_path

    current_request_id = await ipc_state.get_next_request_id()
    response_channel_id = ipc_state.response_channel_id or current_request_id
    response_queue: asyncio.Queue[ResponseDeltaDict] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    ipc_state.active_request_queues[current_request_id] = QueueRegistration(
        loop=loop,
        queue=response_queue,
    )

    fanout_counts: list[int] = []
    final_candidate_counts: list[int] = []
    prompt_payloads: list[dict[str, Any]] = []

    for prompt_text in prompts:
        effective_prompt = prompt_text
        if request.apply_chat_template:
            try:
                effective_prompt = formatter.apply_template(
                    [{"role": "user", "content": prompt_text}]
                )
            except Exception as exc:  # pragma: no cover - template errors
                logger.exception("Failed to apply chat template: %s", exc)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to render prompt template.",
                ) from exc

        prompt_bytes = effective_prompt.encode("utf-8")
        layout_segments = [
            {
                "type": "text",
                "length": len(prompt_bytes),
            }
        ]

        rng_seed = random.randint(0, 2**32 - 1)
        normalized_top_k = request.top_k if request.top_k > 0 else -1
        num_candidates = request.best_of if request.best_of is not None else request.n
        num_candidates = max(1, num_candidates)
        final_candidates = max(1, request.n)
        if final_candidates > num_candidates:
            final_candidates = num_candidates

        prompt_payloads.append(
            {
                "prompt_bytes": prompt_bytes,
                "image_buffers": [],
                "layout": layout_segments,
                "sampling_params": {
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "top_k": normalized_top_k,
                    "min_p": request.min_p,
                    "rng_seed": rng_seed,
                },
                "logits_params": {
                    "top_logprobs": request.logprobs or 0,
                    "frequency_penalty": 0.0,
                    "logit_bias": {},
                    "presence_penalty": 0.0,
                    "repetition_context_size": 60,
                    "repetition_penalty": 1.0,
                },
                "max_generated_tokens": request.max_completion_tokens,
                "stop_sequences": [],
                "tool_schemas_json": "",
                "response_format_json": "",
                "num_candidates": num_candidates,
                "best_of": num_candidates,
                "final_candidates": final_candidates,
                "tool_calling_tokens": {
                    "call_start": formatter.control_tokens.tool_calling.call_start,
                    "call_end": formatter.control_tokens.tool_calling.call_end,
                    "section_start": formatter.control_tokens.tool_calling.section_start,
                    "section_end": formatter.control_tokens.tool_calling.section_end,
                    "name_separator": formatter.control_tokens.tool_calling.name_separator,
                },
            }
        )
        fanout_counts.append(num_candidates)
        final_candidate_counts.append(final_candidates)

    if not prompt_payloads:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one prompt is required.",
        )

    try:
        request_bytes = _build_request_payload(
            request_id=current_request_id,
            model_id=canonical_id,
            model_path=model_path,
            request_type="generation",
            response_channel_id=response_channel_id,
            prompts=prompt_payloads,
        )

        socket = ipc_state.request_socket
        if socket is None:
            raise RuntimeError("Request socket is not initialized.")

        await socket.asend(request_bytes)
        logger.info(
            "Submitted completions request %d with %d prompts.",
            current_request_id,
            len(prompt_payloads),
        )

        response_data = await gather_non_streaming_batch_response(
            current_request_id,
            response_queue,
            fanout_counts,
            final_candidate_counts,
        )

        completion_choices = _convert_chat_choices(response_data["choices"])
        usage = CompletionUsage(
            input_tokens=response_data["prompt_tokens"],
            output_tokens=response_data["completion_tokens"],
            total_tokens=response_data["prompt_tokens"]
            + response_data["completion_tokens"],
        )
        response = CompletionResponse(
            model=request.model,
            choices=completion_choices,
            usage=usage,
        )
        logger.info("Completion request %d completed successfully.", current_request_id)
        return response
    except HTTPException:
        raise
    except InferenceError as exc:
        logger.error(
            "Inference error processing completion request %d: %s",
            current_request_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception(
            "Unexpected error while processing completion request %d: %s",
            current_request_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during completion.",
        ) from exc
    finally:
        binding = ipc_state.active_request_queues.pop(current_request_id, None)
        if binding is not None:
            queue = binding.queue
            try:
                while True:
                    leftover = queue.get_nowait()
                    release_delta_resources(leftover)
                    queue.task_done()
            except asyncio.QueueEmpty:
                pass


def _normalize_prompt_inputs(prompt: str | list[str]) -> list[str]:
    if isinstance(prompt, str):
        return [prompt]
    if isinstance(prompt, list) and prompt:
        if not all(isinstance(entry, str) for entry in prompt):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="All prompt entries must be strings.",
            )
        return prompt
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Invalid prompt format. Expecting string or non-empty list of strings.",
    )


def _convert_chat_choices(
    chat_choices: list[ChatCompletionChoice],
) -> list[CompletionChoice]:
    completion_choices: list[CompletionChoice] = []
    for choice in chat_choices:
        message_content = ""
        if choice.message and choice.message.content is not None:
            message_content = str(choice.message.content)
        completion_choices.append(
            CompletionChoice(
                index=choice.index,
                text=message_content,
                logprobs=_extract_logprob_values(choice.logprobs),
                finish_reason=choice.finish_reason,
            )
        )
    return completion_choices


def _extract_logprob_values(
    logprob_content: ChatCompletionLogProbs.LogProbsContent | None,
) -> list[float] | None:
    if not logprob_content or not logprob_content.content:
        return None
    values = []
    for entry in logprob_content.content:
        try:
            values.append(float(entry.logprob))
        except (TypeError, ValueError):
            continue
    return values or None
