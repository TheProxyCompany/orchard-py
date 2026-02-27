from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, Field

from orchard.ipc.utils import ResponseDeltaDict
from orchard.server.models.reasoning import (
    DEFAULT_BOOLEAN_REASONING_EFFORT,
)
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
    ResponseFailedEvent,
    ResponseIncompleteEvent,
    ResponseInProgressEvent,
    ResponseObject,
    ResponseRequest,
    ResponseStreamState,
    ResponseUsage,
    ResponseCreatedEvent,
    StreamErrorEvent,
    StreamingEvent,
    get_current_timestamp,
    generate_response_id,
)


class DoneEvent(BaseModel):
    type: Literal["done"] = "done"


ResponseEvent: TypeAlias = StreamingEvent | DoneEvent


class ResponsesRequest(ResponseRequest):
    model: str = Field(default="", exclude=True)
    stream: bool = False
    parallel_tool_calls: bool = False

    @classmethod
    def from_text(cls, text: str, **kwargs: Any) -> ResponsesRequest:
        return cls(input=text, **kwargs)

    def to_messages(self) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for item in self.get_message_items():
            message: dict[str, Any] = {
                "role": item.role,
                "content": item.content,
            }
            if item.tool_calls:
                message["tool_calls"] = [
                    tool_call.model_dump(exclude_none=True)
                    for tool_call in item.tool_calls
                ]
            if item.tool_call_id:
                message["tool_call_id"] = item.tool_call_id
            messages.append(message)
        return messages

    def to_submit_kwargs(self) -> dict[str, Any]:
        tools_payload = [
            normalize_response_tool_schema(tool)
            for tool in (self.tools or [])
        ]
        response_format = self.text.to_dict() if self.text is not None else None

        kwargs = {
            "instructions": self.instructions,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "max_generated_tokens": self.max_output_tokens
            if self.max_output_tokens is not None
            else 0,
            "top_logprobs": self.top_logprobs if self.top_logprobs is not None else 0,
            "tools": tools_payload,
            "tool_choice": _tool_choice_to_payload(self.tool_choice),
            "response_format": response_format,
            "task_name": self.task,
            "reasoning_effort": _reasoning_effort_from_request(self),
        }
        return {key: value for key, value in kwargs.items() if value is not None}


@dataclass
class _AggregatedOutputItem:
    item_type: str
    content: str = ""
    arguments: str = ""
    identifier: str = ""


def _reasoning_effort_from_request(request: ResponsesRequest) -> str | None:
    reasoning = request.reasoning
    if reasoning is None:
        return None
    normalized_effort = getattr(reasoning, "normalized_effort", None)
    if normalized_effort is not None:
        return normalized_effort
    return DEFAULT_BOOLEAN_REASONING_EFFORT if isinstance(reasoning, bool) else None


def _tool_choice_to_payload(tool_choice: Any) -> str | dict[str, Any]:
    if tool_choice is None:
        return "auto"
    if hasattr(tool_choice, "to_dict"):
        return tool_choice.to_dict()
    if isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, dict):
        return tool_choice
    return "auto"


def normalize_response_tool_schema(tool: Any) -> Any:
    payload = tool
    if hasattr(tool, "to_dict"):
        payload = tool.to_dict()
    elif hasattr(tool, "model_dump"):
        payload = tool.model_dump(exclude_none=True)

    if not isinstance(payload, dict):
        return payload

    type_name = payload.get("type")
    name = payload.get("name")
    parameters = payload.get("parameters")

    if type_name == "function" and name and parameters is not None:
        description = payload.get("description") or name
        strict = payload.get("strict", True)
        return {
            "name": name,
            "type": "object",
            "description": description,
            "properties": {
                "name": {"const": name},
                "arguments": parameters,
            },
            "strict": strict,
            "required": ["name", "arguments"],
        }

    return payload


def finish_reason_to_incomplete(reason: str | None) -> IncompleteDetails | None:
    normalized = (reason or "").strip().lower()
    if normalized in {"length", "max_tokens", "max_output_tokens"}:
        return IncompleteDetails(reason="max_output_tokens")
    if normalized == "content_filter":
        return IncompleteDetails(reason="content_filter")
    return None


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _value_to_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return str(value)


def _update_usage_from_delta(
    delta: ResponseDeltaDict,
    usage: ResponseUsage,
) -> None:
    usage_payload = delta.get("usage")
    usage_dict = usage_payload if isinstance(usage_payload, dict) else {}

    prompt_tokens = _coerce_int(delta.get("prompt_token_count"))
    if prompt_tokens is None:
        prompt_tokens = _coerce_int(
            usage_dict.get("input_tokens", usage_dict.get("prompt_tokens"))
        )
    if prompt_tokens is not None:
        usage.input_tokens = max(usage.input_tokens, prompt_tokens)

    output_tokens = _coerce_int(delta.get("generation_len"))
    if output_tokens is None:
        output_tokens = _coerce_int(
            usage_dict.get("output_tokens", usage_dict.get("completion_tokens"))
        )
    if output_tokens is not None:
        usage.output_tokens = max(usage.output_tokens, output_tokens)

    cached_tokens = _coerce_int(delta.get("cached_token_count"))
    if cached_tokens is None:
        cached_tokens = _coerce_int(usage_dict.get("cached_tokens"))
    if cached_tokens is not None:
        usage.input_tokens_details = InputTokensDetails(cached_tokens=cached_tokens)

    reasoning_tokens = _coerce_int(delta.get("reasoning_tokens"))
    if reasoning_tokens is None:
        reasoning_tokens = _coerce_int(usage_dict.get("reasoning_tokens"))
    if reasoning_tokens is not None:
        usage.output_tokens_details = OutputTokensDetails(
            reasoning_tokens=reasoning_tokens
        )

    usage.total_tokens = usage.input_tokens + usage.output_tokens


def _process_state_event_for_output(
    event: dict[str, Any],
    output_items: dict[int, _AggregatedOutputItem],
) -> None:
    item_type = str(event.get("item_type") or "message")
    output_index = int(event.get("output_index", 0))
    identifier = str(event.get("identifier", ""))
    event_type = str(event.get("event_type", ""))

    if output_index not in output_items:
        output_items[output_index] = _AggregatedOutputItem(
            item_type=item_type,
            identifier=identifier,
        )

    item = output_items[output_index]

    if item.item_type == "tool_call" and identifier == "arguments":
        if event_type == "content_delta":
            item.arguments += str(event.get("delta", ""))
        elif event_type == "item_completed" and "value" in event:
            item.arguments = _value_to_string(event["value"])
        return

    if event_type == "content_delta":
        item.content += str(event.get("delta", ""))
    elif event_type == "item_completed":
        if item.item_type == "tool_call":
            item.identifier = identifier
        elif "value" in event:
            item.content = _value_to_string(event["value"])


def _build_output_items(
    output_items: dict[int, _AggregatedOutputItem],
) -> list[OutputMessage | OutputFunctionCall | OutputReasoning]:
    result: list[OutputMessage | OutputFunctionCall | OutputReasoning] = []

    for output_index in sorted(output_items.keys()):
        item = output_items[output_index]
        if item.item_type == "message":
            result.append(
                OutputMessage(
                    status=OutputStatus.COMPLETED,
                    content=[OutputTextContent(text=item.content)] if item.content else [],
                )
            )
        elif item.item_type == "tool_call":
            function_name = item.identifier.removeprefix("tool_call:")
            result.append(
                OutputFunctionCall(
                    name=function_name,
                    arguments=item.arguments,
                    status=OutputStatus.COMPLETED,
                )
            )
        elif item.item_type == "reasoning":
            result.append(
                OutputReasoning(
                    status=OutputStatus.COMPLETED,
                    content=[ReasoningContent(text=item.content)] if item.content else [],
                )
            )

    if not result:
        result.append(OutputMessage(status=OutputStatus.COMPLETED, content=[]))

    return result


def aggregate_non_streaming_response(
    deltas: list[ResponseDeltaDict],
    model_id: str,
    request: ResponsesRequest,
) -> ResponseObject:
    created_at = get_current_timestamp()
    completed_at: int | None = None
    output_items: dict[int, _AggregatedOutputItem] = {}
    fallback_content = ""
    usage = ResponseUsage(input_tokens=0, output_tokens=0, total_tokens=0)
    error_detail: str | None = None
    finish_reason: str | None = None

    for delta in deltas:
        error = delta.get("error")
        if error:
            error_detail = str(error)

        content = delta.get("content")
        if isinstance(content, str):
            fallback_content += content

        for state_event in delta.get("state_events") or []:
            if isinstance(state_event, dict):
                _process_state_event_for_output(state_event, output_items)

        _update_usage_from_delta(delta, usage)

        if finish := delta.get("finish_reason"):
            finish_reason = str(finish).lower()

        if delta.get("is_final_delta", False):
            completed_at = get_current_timestamp()
            break

    if error_detail:
        raise RuntimeError(error_detail)

    incomplete_details = finish_reason_to_incomplete(finish_reason)

    if not output_items and fallback_content:
        output: list[OutputMessage | OutputFunctionCall | OutputReasoning] = [
            OutputMessage(
                status=OutputStatus.COMPLETED,
                content=[OutputTextContent(text=fallback_content)],
            )
        ]
    else:
        output = _build_output_items(output_items)

    return ResponseObject(
        model=model_id,
        created_at=created_at,
        completed_at=completed_at,
        output=output,
        usage=usage,
        status=OutputStatus.INCOMPLETE
        if incomplete_details
        else OutputStatus.COMPLETED,
        incomplete_details=incomplete_details,
        metadata=request.metadata,
        parallel_tool_calls=request.parallel_tool_calls,
        temperature=request.temperature,
        top_p=request.top_p,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
        top_k=request.top_k,
        min_p=request.min_p,
        truncation=request.truncation,
        instructions=request.instructions,
        max_output_tokens=request.max_output_tokens,
        top_logprobs=request.top_logprobs,
        tool_choice=request.tool_choice,
        tools=request.tools or [],
        max_tool_calls=request.max_tool_calls,
        text=request.text,
    )


def _process_state_event_for_streaming(
    event: dict[str, Any],
    stream_state: ResponseStreamState,
) -> list[ResponseEvent]:
    mapped_events: list[ResponseEvent] = []

    event_type = str(event.get("event_type", ""))
    item_type = str(event.get("item_type") or "message")
    output_index = int(event.get("output_index", 0))
    identifier = str(event.get("identifier", ""))

    item = stream_state.get_or_create_item(output_index, item_type, identifier)

    if item_type == "tool_call" and identifier == "arguments":
        if event_type == "content_delta":
            item.accumulated_arguments += str(event.get("delta", ""))
            mapped_events.append(
                FunctionCallArgumentsDeltaEvent(
                    sequence_number=stream_state.next_sequence_number(),
                    item_id=item.item_id,
                    output_index=output_index,
                    delta=str(event.get("delta", "")),
                )
            )
        elif event_type == "item_completed":
            if "value" in event:
                item.accumulated_arguments = _value_to_string(event["value"])
            mapped_events.append(
                FunctionCallArgumentsDoneEvent(
                    sequence_number=stream_state.next_sequence_number(),
                    item_id=item.item_id,
                    output_index=output_index,
                    arguments=item.accumulated_arguments,
                )
            )
        return mapped_events

    if event_type == "item_started":
        mapped_events.append(
            OutputItemAddedEvent(
                sequence_number=stream_state.next_sequence_number(),
                output_index=output_index,
                item=item.to_skeleton(),
            )
        )
        if item_type == "message":
            mapped_events.append(
                ContentPartAddedEvent(
                    sequence_number=stream_state.next_sequence_number(),
                    item_id=item.item_id,
                    output_index=output_index,
                    content_index=0,
                    part=OutputTextContent(text=""),
                )
            )
        elif item_type == "reasoning":
            mapped_events.append(
                ContentPartAddedEvent(
                    sequence_number=stream_state.next_sequence_number(),
                    item_id=item.item_id,
                    output_index=output_index,
                    content_index=0,
                    part=ReasoningContent(text=""),
                )
            )
        return mapped_events

    if event_type == "content_delta":
        delta_text = str(event.get("delta", ""))
        item.accumulated_content += delta_text
        if item_type == "message":
            mapped_events.append(
                OutputTextDeltaEvent(
                    sequence_number=stream_state.next_sequence_number(),
                    item_id=item.item_id,
                    output_index=output_index,
                    content_index=0,
                    delta=delta_text,
                )
            )
        elif item_type == "reasoning":
            mapped_events.append(
                ReasoningDeltaEvent(
                    sequence_number=stream_state.next_sequence_number(),
                    item_id=item.item_id,
                    output_index=output_index,
                    content_index=0,
                    delta=delta_text,
                )
            )
        return mapped_events

    if event_type == "item_completed":
        item.status = OutputStatus.COMPLETED
        if item_type == "tool_call":
            item.function_name = identifier.removeprefix("tool_call:")
        elif "value" in event:
            item.accumulated_content = _value_to_string(event["value"])

        if item_type == "message":
            mapped_events.append(
                OutputTextDoneEvent(
                    sequence_number=stream_state.next_sequence_number(),
                    item_id=item.item_id,
                    output_index=output_index,
                    content_index=0,
                    text=item.accumulated_content,
                )
            )
            mapped_events.append(
                ContentPartDoneEvent(
                    sequence_number=stream_state.next_sequence_number(),
                    item_id=item.item_id,
                    output_index=output_index,
                    content_index=0,
                    part=OutputTextContent(text=item.accumulated_content),
                )
            )
        elif item_type == "reasoning":
            mapped_events.append(
                ReasoningDoneEvent(
                    sequence_number=stream_state.next_sequence_number(),
                    item_id=item.item_id,
                    output_index=output_index,
                    content_index=0,
                    text=item.accumulated_content,
                )
            )

        mapped_events.append(
            OutputItemDoneEvent(
                sequence_number=stream_state.next_sequence_number(),
                output_index=output_index,
                item=item.to_completed(),
            )
        )

    return mapped_events


def _emit_stream_fallback_item_done(
    stream_state: ResponseStreamState,
) -> list[ResponseEvent]:
    mapped_events: list[ResponseEvent] = []

    for output_index, item in sorted(stream_state.items.items()):
        if item.status == OutputStatus.COMPLETED:
            continue

        item.status = OutputStatus.COMPLETED

        if item.item_type == "message":
            mapped_events.append(
                OutputTextDoneEvent(
                    sequence_number=stream_state.next_sequence_number(),
                    item_id=item.item_id,
                    output_index=output_index,
                    content_index=0,
                    text=item.accumulated_content,
                )
            )
            mapped_events.append(
                ContentPartDoneEvent(
                    sequence_number=stream_state.next_sequence_number(),
                    item_id=item.item_id,
                    output_index=output_index,
                    content_index=0,
                    part=OutputTextContent(text=item.accumulated_content),
                )
            )
        elif item.item_type == "tool_call":
            mapped_events.append(
                FunctionCallArgumentsDoneEvent(
                    sequence_number=stream_state.next_sequence_number(),
                    item_id=item.item_id,
                    output_index=output_index,
                    arguments=item.accumulated_arguments,
                )
            )
        elif item.item_type == "reasoning":
            mapped_events.append(
                ReasoningDoneEvent(
                    sequence_number=stream_state.next_sequence_number(),
                    item_id=item.item_id,
                    output_index=output_index,
                    content_index=0,
                    text=item.accumulated_content,
                )
            )

        mapped_events.append(
            OutputItemDoneEvent(
                sequence_number=stream_state.next_sequence_number(),
                output_index=output_index,
                item=item.to_completed(),
            )
        )

    return mapped_events


async def iter_response_events(
    delta_iterator: AsyncIterator[ResponseDeltaDict],
    model_id: str,
    response_id: str | None = None,
) -> AsyncIterator[ResponseEvent]:
    stream_state = ResponseStreamState(
        response_id=response_id or generate_response_id(),
        model=model_id,
    )

    yield ResponseCreatedEvent(
        sequence_number=stream_state.next_sequence_number(),
        response=stream_state.snapshot(),
    )
    yield ResponseInProgressEvent(
        sequence_number=stream_state.next_sequence_number(),
        response=stream_state.snapshot(),
    )

    error_detail: str | None = None
    finish_reason: str | None = None
    usage = ResponseUsage(input_tokens=0, output_tokens=0, total_tokens=0)

    async for delta in delta_iterator:
        if error := delta.get("error"):
            error_detail = str(error)
            break

        for event in delta.get("state_events") or []:
            if isinstance(event, dict):
                for mapped_event in _process_state_event_for_streaming(event, stream_state):
                    yield mapped_event

        _update_usage_from_delta(delta, usage)
        if usage.total_tokens > 0:
            stream_state.usage = usage

        if finish := delta.get("finish_reason"):
            finish_reason = str(finish).lower()

        if delta.get("is_final_delta", False):
            break

    for mapped_event in _emit_stream_fallback_item_done(stream_state):
        yield mapped_event

    if error_detail:
        stream_state.status = OutputStatus.FAILED
        yield ResponseFailedEvent(
            sequence_number=stream_state.next_sequence_number(),
            response=stream_state.snapshot(),
        )
        yield StreamErrorEvent(
            sequence_number=stream_state.next_sequence_number(),
            error={"message": error_detail, "type": "inference_error"},
        )
    else:
        stream_state.completed_at = get_current_timestamp()
        incomplete_details = finish_reason_to_incomplete(finish_reason)
        if incomplete_details is not None:
            stream_state.status = OutputStatus.INCOMPLETE
            stream_state.incomplete_details = incomplete_details
            yield ResponseIncompleteEvent(
                sequence_number=stream_state.next_sequence_number(),
                response=stream_state.snapshot(),
            )
        else:
            stream_state.status = OutputStatus.COMPLETED
            yield ResponseCompletedEvent(
                sequence_number=stream_state.next_sequence_number(),
                response=stream_state.snapshot(),
            )

    yield DoneEvent()


__all__ = [
    "DoneEvent",
    "ResponseEvent",
    "ResponsesRequest",
    "aggregate_non_streaming_response",
    "finish_reason_to_incomplete",
    "iter_response_events",
    "normalize_response_tool_schema",
]
