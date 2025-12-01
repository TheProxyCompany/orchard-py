from __future__ import annotations

from dataclasses import dataclass

from proxy_inference_engine.server.models.chat.format import (
    ChatCompletionJsonObjectResponseFormat,
    ChatCompletionJSONSchemaResponseFormat,
    ChatCompletionTextResponseFormat,
)
from proxy_inference_engine.server.models.chat.request import (
    ChatCompletionRequest,
    ChatMessage,
)
from proxy_inference_engine.server.models.chat.tools import ChatCompletionTool
from proxy_inference_engine.server.models.reasoning import ReasoningEffort


@dataclass
class NormalizedChatInstance:
    prompt_index: int
    messages: list[ChatMessage]
    max_completion_tokens: int | None
    temperature: float
    top_p: float | None
    top_k: int | None
    min_p: float | None
    logprobs: bool
    top_logprobs: int | None
    tools: list[ChatCompletionTool] | None
    response_format: (
        ChatCompletionTextResponseFormat
        | ChatCompletionJSONSchemaResponseFormat
        | ChatCompletionJsonObjectResponseFormat
        | None
    )
    stop_sequences: list[str]
    best_of: int
    final_candidates: int
    task: str | None = None
    reasoning_effort: ReasoningEffort | None = None


def normalize_chat_request(
    request: ChatCompletionRequest,
) -> list[NormalizedChatInstance]:
    batch_size = request.batch_size

    max_tokens = request.get_normalized_field("max_completion_tokens")
    temperatures = request.get_normalized_field("temperature")
    top_ps = request.get_normalized_field("top_p")
    top_ks = request.get_normalized_field("top_k")
    min_ps = request.get_normalized_field("min_p")
    logprobs_flags = request.get_normalized_field("logprobs")
    top_logprobs = request.get_normalized_field("top_logprobs")
    tools_list = request.get_normalized_field("tools")
    response_formats = request.get_normalized_field("response_format")
    stop_sequences = request.get_normalized_field("stop")
    task_values = request.get_normalized_field("task")
    candidate_counts = request.get_normalized_field("n")
    best_of_counts = request.get_normalized_field("best_of")
    reasoning_efforts = request.get_normalized_field("reasoning_effort")

    instances: list[NormalizedChatInstance] = []
    for idx in range(batch_size):
        temp_value = temperatures[idx] if temperatures[idx] is not None else 1.0
        top_p_value = top_ps[idx] if top_ps[idx] is not None else 1.0
        top_k_value = top_ks[idx]
        min_p_value = min_ps[idx] if min_ps[idx] is not None else 0.0
        logprobs_enabled = (
            bool(logprobs_flags[idx]) if logprobs_flags[idx] is not None else False
        )
        top_logprobs_value = top_logprobs[idx] if logprobs_enabled else 0
        tools_value = tools_list[idx] if tools_list[idx] else None
        response_format_value = response_formats[idx]
        stops_value = stop_sequences[idx] if stop_sequences[idx] else []
        candidate_count = (
            int(candidate_counts[idx]) if candidate_counts[idx] is not None else 1
        )
        if candidate_count < 1:
            raise ValueError("n must be at least 1 for all batch entries.")
        best_of_value = (
            int(best_of_counts[idx])
            if best_of_counts[idx] is not None
            else candidate_count
        )
        messages = request.messages[idx]
        if not isinstance(messages, list):
            messages = [messages]

        instances.append(
            NormalizedChatInstance(
                prompt_index=idx,
                messages=messages,
                max_completion_tokens=max_tokens[idx],
                temperature=float(temp_value),
                top_p=float(top_p_value) if top_p_value is not None else None,
                top_k=int(top_k_value) if top_k_value is not None else None,
                min_p=float(min_p_value) if min_p_value is not None else None,
                logprobs=logprobs_enabled,
                top_logprobs=int(top_logprobs_value) if top_logprobs_value else 0,
                tools=tools_value,
                response_format=response_format_value,
                stop_sequences=list(stops_value),
                best_of=best_of_value,
                final_candidates=candidate_count,
                task=task_values[idx],
                reasoning_effort=reasoning_efforts[idx],
            )
        )

    return instances


__all__ = ["NormalizedChatInstance", "normalize_chat_request"]
