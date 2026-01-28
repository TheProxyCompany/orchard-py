import asyncio
import json
import logging
import random
from collections import defaultdict
from collections.abc import AsyncIterable
from contextlib import AsyncExitStack
from typing import Any

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from orchard.server.routes._common import (
    _ModelNotReadyError,
    managed_stream_session,
    resolve_model,
)
from orchard.ipc.serialization import _build_request_payload
from orchard.ipc.utils import (
    ResponseDeltaDict,
    normalise_delta_payload,
    release_delta_resources,
)
from orchard.server.dependencies import IPCStateDep, ModelRegistryDep
from orchard.server.exceptions import InferenceError
from orchard.server.models.chat import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
)
from orchard.server.models.chat.logprobs import ChatCompletionLogProbs
from orchard.server.models.chat.output import (
    generate_chat_completion_id,
    get_current_timestamp,
)
from orchard.server.utils.batching import normalize_chat_request

logger = logging.getLogger(__name__)

chat_router = APIRouter()


def _dedupe_stop_sequences(raw_stop: list[str] | str | None) -> list[str]:
    if not raw_stop:
        return []

    if isinstance(raw_stop, str):
        candidates = [raw_stop]
    else:
        candidates = [item for item in raw_stop if item]

    seen: set[str] = set()
    unique: list[str] = []
    for seq in candidates:
        if seq not in seen:
            seen.add(seq)
            unique.append(seq)
    return unique


@chat_router.post(
    "/chat/completions",
    summary="Create a chat completion",
    tags=["Chat"],
    response_model=None,  # Disable automatic response model to support streaming
)
async def handle_completion_request(
    request: ChatCompletionRequest,
    ipc_state: IPCStateDep,
    model_registry: ModelRegistryDep,
) -> ChatCompletionResponse | EventSourceResponse | JSONResponse:
    """
    Handles requests to the `/v1/chat/completions` endpoint.
    """
    logger.info("Chat completion request for model: %s", request.model)

    try:
        canonical_id, model_info = await resolve_model(model_registry, request.model)
    except _ModelNotReadyError as exc:
        return exc.response

    formatter = model_info.formatter
    model_path = model_info.model_path

    try:
        normalized_instances = normalize_chat_request(request)
    except ValueError as exc:
        logger.error("Validation error normalizing batch parameters: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    batch_size = len(normalized_instances)
    logger.debug("Normalized chat batch size: %d", batch_size)

    fanout_counts = [instance.best_of for instance in normalized_instances]
    final_candidate_counts = [
        instance.final_candidates for instance in normalized_instances
    ]
    total_expected_sequences = sum(fanout_counts)
    if total_expected_sequences <= 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one candidate must be requested.",
        )

    if batch_size == 0:
        logger.error("No prompts provided.")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one prompt is required.",
        )

    prompt_payloads: list[dict[str, Any]] = []
    for instance in normalized_instances:
        messages_as_dicts = [
            msg.model_dump(exclude_none=True) for msg in instance.messages
        ]
        prompt_text = formatter.apply_template(
            messages_as_dicts, reasoning=instance.reasoning_effort is not None
        )
        logger.info("Prompt text: %s", prompt_text)
        stop_sequences = _dedupe_stop_sequences(instance.stop_sequences)

        tools_payload = (
            [tool.to_dict() for tool in instance.tools] if instance.tools else None
        )
        tool_schemas_str = json.dumps(tools_payload) if tools_payload else ""
        response_format_str = (
            json.dumps(instance.response_format.to_dict())
            if instance.response_format
            else ""
        )
        prompt_bytes = prompt_text.encode("utf-8")

        payload: dict[str, Any] = {
            "prompt": prompt_text,
            "sampling_params": {
                "temperature": instance.temperature,
                "top_p": instance.top_p if instance.top_p is not None else 1.0,
                "top_k": instance.top_k if instance.top_k is not None else -1,
                "min_p": instance.min_p if instance.min_p is not None else 0.0,
                "rng_seed": random.randint(0, 2**32 - 1),
            },
            "logits_params": {
                "top_logprobs": instance.top_logprobs,
                "frequency_penalty": 0.0,
                "logit_bias": {},
                "presence_penalty": 0.0,
                "repetition_context_size": 60,
                "repetition_penalty": 1.0,
            },
            "max_generated_tokens": instance.max_completion_tokens or 1024,
            "stop_sequences": stop_sequences,
            "tool_schemas_json": tool_schemas_str,
            "response_format_json": response_format_str,
            "image_buffers": [],
            "layout": [
                {
                    "type": "text",
                    "length": len(prompt_bytes),
                }
            ],
            "num_candidates": instance.best_of,
            "best_of": instance.best_of,
            "final_candidates": instance.final_candidates,
        }
        if instance.task is not None:
            payload["task"] = instance.task
        if instance.reasoning_effort is not None:
            payload["reasoning_effort"] = instance.reasoning_effort

        # Pass tool_calling tokens from model profile
        tc = formatter.control_tokens.tool_calling
        payload["tool_calling_tokens"] = {
            "call_start": tc.call_start,
            "call_end": tc.call_end,
            "section_start": tc.section_start,
            "section_end": tc.section_end,
            "name_separator": tc.name_separator,
        }
        prompt_payloads.append(payload)

    current_request_id = await ipc_state.get_next_request_id()
    logger.debug("Generated request ID: %d", current_request_id)
    response_channel_id = ipc_state.response_channel_id or current_request_id

    response_queue = asyncio.Queue[ResponseDeltaDict]()
    exit_stack = AsyncExitStack()
    await exit_stack.enter_async_context(
        managed_stream_session(ipc_state, current_request_id, response_queue)
    )

    try:
        logger.debug("Submitting batched request %d to C++ engine", current_request_id)
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
        logger.info("Submitted batched request %d successfully", current_request_id)
        if request.stream:
            logger.debug("Handling streaming for request %d", current_request_id)

            async def event_stream() -> AsyncIterable[dict[str, str]]:
                try:
                    async for chunk in stream_response_generator(
                        current_request_id,
                        response_queue,
                        ipc_state,
                        request.model,
                        total_expected_sequences,
                    ):
                        yield chunk
                finally:
                    await exit_stack.aclose()

            return EventSourceResponse(
                content=event_stream(),
                media_type="text/event-stream",
            )

        logger.debug("Handling non-streaming for request %d", current_request_id)
        response_data = await gather_non_streaming_batch_response(
            current_request_id,
            response_queue,
            fanout_counts,
            final_candidate_counts,
        )
        usage = ChatCompletionUsage(
            input_tokens=response_data["prompt_tokens"],
            output_tokens=response_data["completion_tokens"],
            total_tokens=response_data["prompt_tokens"]
            + response_data["completion_tokens"],
        )
        final_response = ChatCompletionResponse(
            id=generate_chat_completion_id(),
            created=get_current_timestamp(),
            model=request.model,
            choices=response_data["choices"],
            usage=usage,
        )
        logger.info("Non-streaming request %d successful.", current_request_id)
        return final_response
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error submitting request: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during completion.",
        ) from e
    finally:
        await exit_stack.aclose()


async def gather_non_streaming_batch_response(
    request_id: int,
    queue: asyncio.Queue[ResponseDeltaDict],
    prompt_fanout_counts: list[int],
    prompt_final_counts: list[int],
) -> dict[str, Any]:
    """Collects deltas for all sequences belonging to a parent request."""

    total_expected = sum(prompt_fanout_counts)
    prompt_states: list[list[dict[str, Any]]] = [
        [
            {
                "content": "",
                "finish_reason": "unknown",
                "completion_tokens": 0,
                "logprobs_entries": [],
                "sequence_id": None,
                "completed": False,
                "error_message": None,
                "cumulative_logprob": None,
                "generation_len": 0,
            }
            for _ in range(candidate_count)
        ]
        for candidate_count in prompt_fanout_counts
    ]
    prompt_token_totals: list[int] = [0 for _ in prompt_fanout_counts]

    remaining_sequences = total_expected

    while remaining_sequences > 0:
        try:
            delta = await asyncio.wait_for(queue.get(), timeout=30.0)
        except TimeoutError as exc:
            logger.error(
                "Timeout waiting for delta for non-streaming request %d", request_id
            )
            raise InferenceError("Timeout receiving response from engine.") from exc
        else:
            try:
                normalise_delta_payload(delta)
                prompt_index = delta.get("prompt_index")
                candidate_index = delta.get("candidate_index")
                sequence_id = delta.get("sequence_id")

                if prompt_index is None or candidate_index is None:
                    logger.warning(
                        "Received delta missing prompt/candidate index for request %d: %s",
                        request_id,
                        delta,
                    )
                    continue

                if not (0 <= prompt_index < len(prompt_states)):
                    logger.warning(
                        "Prompt index %s out of range for request %d",
                        prompt_index,
                        request_id,
                    )
                    continue

                if not (0 <= candidate_index < len(prompt_states[prompt_index])):
                    logger.warning(
                        "Candidate index %s out of range for request %d prompt %d",
                        candidate_index,
                        request_id,
                        prompt_index,
                    )
                    continue

                state = prompt_states[prompt_index][candidate_index]
                if state["sequence_id"] is None and sequence_id is not None:
                    state["sequence_id"] = sequence_id

                prompt_token_count = delta.get("prompt_token_count", 0)
                if prompt_token_count > 0 and prompt_token_totals[prompt_index] == 0:
                    prompt_token_totals[prompt_index] = prompt_token_count

                delta_content = delta.get("content") or ""
                if delta_content:
                    state["content"] += delta_content

                tokens = delta.get("tokens", [])
                state["completion_tokens"] += len(tokens)

                if top_logprobs_data := delta.get("top_logprobs"):
                    top_logprobs_list: list[ChatCompletionLogProbs] = []
                    for item in top_logprobs_data:
                        token_val = (
                            (item.get("token") or "") if isinstance(item, dict) else ""
                        )
                        try:
                            raw_logprob = item.get("logprob", item.get("probability"))
                            if not isinstance(raw_logprob, (int | float)):
                                continue
                            logprob_val = round(float(raw_logprob), 6)
                        except (TypeError, ValueError):
                            logger.debug(
                                "Skipping top_logprobs entry without numeric logprob for request %d: %s",
                                request_id,
                                item,
                            )
                            continue

                        top_logprobs_list.append(
                            ChatCompletionLogProbs(
                                token=token_val,
                                logprob=logprob_val,
                                bytes=list(token_val.encode("utf-8")),
                            )
                        )

                    chosen_token_str = delta_content
                    chosen_token_logprob = -999.0
                    for item in top_logprobs_data:
                        if (
                            isinstance(item, dict)
                            and item.get("token") == chosen_token_str
                            and (
                                raw_logprob := item.get(
                                    "logprob", item.get("probability")
                                )
                            )
                            is not None
                        ):
                            if isinstance(raw_logprob, (int | float)):
                                chosen_token_logprob = float(raw_logprob)
                                break

                    logprob_entry = ChatCompletionLogProbs(
                        token=chosen_token_str,
                        logprob=chosen_token_logprob,
                        bytes=list(chosen_token_str.encode("utf-8")),
                        top_logprobs=top_logprobs_list,
                    )
                    state["logprobs_entries"].append(logprob_entry)

                finish_reason = delta.get("finish_reason") or "unknown"
                if finish_reason.lower() == "error" and delta_content:
                    state["error_message"] = delta_content

                if delta.get("is_final_delta", False) and not state["completed"]:
                    state["finish_reason"] = finish_reason
                    state["completed"] = True
                    cumulative_val = delta.get("cumulative_logprob")
                    if cumulative_val is not None:
                        try:
                            state["cumulative_logprob"] = float(cumulative_val)
                        except (TypeError, ValueError):
                            logger.debug(
                                "Failed to parse cumulative_logprob for request %d prompt %d candidate %d: %s",
                                request_id,
                                prompt_index,
                                candidate_index,
                                cumulative_val,
                            )
                    generation_len_val = delta.get("generation_len")
                    if generation_len_val is not None:
                        try:
                            state["generation_len"] = int(generation_len_val)
                        except (TypeError, ValueError):
                            logger.debug(
                                "Failed to parse generation_len for request %d prompt %d candidate %d: %s",
                                request_id,
                                prompt_index,
                                candidate_index,
                                generation_len_val,
                            )
                    remaining_sequences -= 1
                    logger.debug(
                        "Sequence %s for request %d (prompt %d, candidate %d) completed with reason %s",
                        sequence_id,
                        request_id,
                        prompt_index,
                        candidate_index,
                        finish_reason,
                    )
            finally:
                queue.task_done()
                release_delta_resources(delta)

    total_prompt_tokens = sum(prompt_token_totals)

    total_completion_tokens = sum(
        candidate_state["completion_tokens"]
        for prompt_state in prompt_states
        for candidate_state in prompt_state
    )

    selections: list[list[tuple[float, int, dict[str, Any]]]] = []
    for prompt_idx, candidate_states in enumerate(prompt_states):
        if not candidate_states:
            selections.append([])
            continue
        fanout = (
            prompt_fanout_counts[prompt_idx]
            if prompt_idx < len(prompt_fanout_counts)
            else len(candidate_states)
        )
        final_target = (
            prompt_final_counts[prompt_idx]
            if prompt_idx < len(prompt_final_counts)
            else len(candidate_states)
        )
        if final_target <= 0:
            final_target = 1
        final_target = min(final_target, len(candidate_states))

        scored_candidates: list[tuple[float, int, dict[str, Any]]] = []
        for idx, candidate_state in enumerate(candidate_states):
            cumulative = candidate_state.get("cumulative_logprob")
            generation_len = candidate_state.get("generation_len", 0)
            if cumulative is not None and generation_len and generation_len > 0:
                score = float(cumulative) / float(generation_len)
            else:
                score = float("-inf")
            candidate_state["average_logprob"] = score
            scored_candidates.append((score, idx, candidate_state))

        if final_target < fanout:
            scored_candidates.sort(key=lambda entry: (-entry[0], entry[1]))
            selections.append(scored_candidates[:final_target])
        else:
            selections.append(scored_candidates)

    choices: list[ChatCompletionChoice] = []
    single_prompt = len(prompt_states) == 1
    running_index = 0
    for _, selected_entries in enumerate(selections):
        for rank_within_prompt, (_, _, candidate_state) in enumerate(selected_entries):
            logprobs_object = None
            if candidate_state["logprobs_entries"]:
                logprobs_object = ChatCompletionLogProbs.LogProbsContent(
                    content=candidate_state["logprobs_entries"]
                )

            choice_content = (
                candidate_state["error_message"]
                if candidate_state["error_message"] is not None
                else candidate_state["content"]
            )

            choice_index = rank_within_prompt if single_prompt else running_index
            running_index += 1

            choices.append(
                ChatCompletionChoice(
                    index=choice_index,
                    message=ChatMessage(role="assistant", content=choice_content),
                    finish_reason=candidate_state["finish_reason"],
                    logprobs=logprobs_object,
                )
            )

    return {
        "choices": choices,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
    }


async def stream_response_generator(
    request_id: int,
    queue: asyncio.Queue[ResponseDeltaDict],
    ipc_state: IPCStateDep,
    model_name: str,
    expected_sequences: int,
) -> AsyncIterable[dict[str, str]]:
    """
    Generates Server-Sent Events (SSE) from the response queue.

    This generator handles streaming responses from the C++ engine, converting
    them to OpenAI-compatible SSE format. It ensures proper cleanup of resources
    even in case of errors or client disconnections.
    """
    completion_tokens_by_candidate: defaultdict[tuple[int, int], int] = defaultdict(int)
    completed_sequences: set[int] = set()
    completed_candidate_slots: set[tuple[int, int]] = set()
    remaining_sequences = expected_sequences
    chat_completion_id = generate_chat_completion_id()
    created_at = get_current_timestamp()
    logger.info("Starting SSE stream for request %d", request_id)

    while True:
        try:
            delta_dict = await asyncio.wait_for(queue.get(), timeout=30.0)
        except TimeoutError:
            logger.error(
                "Timeout waiting for delta for streaming request %d", request_id
            )
            break
        else:
            try:
                prompt_index = int(delta_dict.get("prompt_index", 0) or 0)
                candidate_index = int(delta_dict.get("candidate_index", 0) or 0)
                candidate_key = (prompt_index, candidate_index)

                normalise_delta_payload(delta_dict)

                chunk_choice = ChatCompletionChunkChoice(
                    index=candidate_index,
                    delta=ChatMessage(
                        # only send assistant role for the first chunk
                        role="assistant"
                        if completion_tokens_by_candidate[candidate_key] <= 0
                        else None,
                        content=delta_dict.get("content", None),
                    ),
                    finish_reason=None,
                    logprobs=None,
                )
                chunk = ChatCompletionChunk(
                    id=chat_completion_id,
                    created=created_at,
                    model=model_name,
                    choices=[chunk_choice],
                )

                token_list = delta_dict.get("tokens", [])
                completion_tokens_by_candidate[candidate_key] += len(token_list)

                yield {"data": chunk.model_dump_json(exclude_none=True)}

                if delta_dict.get("is_final_delta", False):
                    finish_reason = delta_dict.get("finish_reason", "stop")
                    sequence_id = delta_dict.get("sequence_id")

                    if (
                        sequence_id is not None
                        and sequence_id not in completed_sequences
                    ):
                        completed_sequences.add(sequence_id)
                        remaining_sequences -= 1
                    elif (
                        sequence_id is None
                        and candidate_key not in completed_candidate_slots
                    ):
                        completed_candidate_slots.add(candidate_key)
                        remaining_sequences -= 1

                    final_chunk_choice = ChatCompletionChunkChoice(
                        index=candidate_index,
                        delta=ChatMessage(role=None, content=None),
                        finish_reason=finish_reason,
                        logprobs=None,
                    )
                    final_chunk = ChatCompletionChunk(
                        id=chat_completion_id,
                        created=created_at,
                        model=model_name,
                        choices=[final_chunk_choice],
                    )
                    yield {"data": final_chunk.model_dump_json(exclude_none=True)}
                    if remaining_sequences <= 0:
                        break
            finally:
                queue.task_done()
                release_delta_resources(delta_dict)

    yield {"data": "[DONE]"}
    logger.info("SSE stream for request %d completed", request_id)
