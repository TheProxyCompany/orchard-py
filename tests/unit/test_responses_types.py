from __future__ import annotations

import pytest

from orchard.clients.responses import (
    ResponsesRequest,
    finish_reason_to_incomplete,
    iter_response_events,
)
from orchard.server.models.responses.request import (
    InputFunctionCall,
    InputFunctionCallOutput,
    InputMessageItem,
    InputReasoning,
)


def test_to_messages_with_text_input() -> None:
    request = ResponsesRequest.from_text("Say hello")
    messages = request.to_messages()

    assert messages == [{"role": "user", "content": "Say hello"}]


def test_to_messages_with_message_items() -> None:
    request = ResponsesRequest(
        input=[
            InputMessageItem(role="system", content="You are concise."),
            InputMessageItem(role="user", content="What is 2+2?"),
        ]
    )

    messages = request.to_messages()
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are concise."
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "What is 2+2?"


def test_to_messages_with_function_call_and_output() -> None:
    request = ResponsesRequest(
        input=[
            InputFunctionCall(
                call_id="call_123",
                name="get_weather",
                arguments='{"location":"SF"}',
            ),
            InputFunctionCallOutput(
                call_id="call_123",
                output='{"temperature":65}',
            ),
        ]
    )

    messages = request.to_messages()
    assert len(messages) == 2

    assistant_message = messages[0]
    assert assistant_message["role"] == "assistant"
    assert assistant_message["content"] == ""
    assert "tool_calls" in assistant_message
    assert len(assistant_message["tool_calls"]) == 1
    assert assistant_message["tool_calls"][0]["id"] == "call_123"
    assert assistant_message["tool_calls"][0]["function"]["name"] == "get_weather"
    assert (
        assistant_message["tool_calls"][0]["function"]["arguments"]
        == '{"location":"SF"}'
    )

    tool_message = messages[1]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_123"
    assert tool_message["content"] == '{"temperature":65}'


def test_to_messages_with_mixed_items_skips_reasoning() -> None:
    request = ResponsesRequest(
        input=[
            InputMessageItem(role="user", content="First prompt"),
            InputFunctionCall(
                call_id="call_1",
                name="lookup",
                arguments='{"k":"v"}',
            ),
            InputReasoning(summary=[{"text": "internal"}]),
            InputFunctionCallOutput(call_id="call_1", output='{"ok":true}'),
            InputMessageItem(role="assistant", content="Follow-up"),
        ]
    )

    messages = request.to_messages()
    roles = [message["role"] for message in messages]

    assert roles == ["user", "assistant", "tool", "assistant"]
    assert len(messages) == 4


def test_from_text_convenience_constructor() -> None:
    request = ResponsesRequest.from_text("Hello there", temperature=0.2, stream=True)
    assert request.stream is True
    assert request.temperature == 0.2
    assert request.to_messages() == [{"role": "user", "content": "Hello there"}]


def test_finish_reason_to_incomplete() -> None:
    assert finish_reason_to_incomplete("length") is not None
    assert finish_reason_to_incomplete("max_output_tokens") is not None
    assert finish_reason_to_incomplete("content_filter") is not None
    assert finish_reason_to_incomplete("stop") is None


@pytest.mark.asyncio
async def test_delta_to_event_mapping_streaming() -> None:
    deltas = [
        {
            "request_id": 1,
            "is_final_delta": False,
            "prompt_token_count": 8,
            "generation_len": 1,
            "state_events": [
                {
                    "event_type": "item_started",
                    "item_type": "message",
                    "output_index": 0,
                    "identifier": "",
                    "delta": "",
                },
                {
                    "event_type": "content_delta",
                    "item_type": "message",
                    "output_index": 0,
                    "identifier": "",
                    "delta": "Hel",
                },
            ],
        },
        {
            "request_id": 1,
            "is_final_delta": True,
            "finish_reason": "stop",
            "generation_len": 2,
            "state_events": [
                {
                    "event_type": "content_delta",
                    "item_type": "message",
                    "output_index": 0,
                    "identifier": "",
                    "delta": "lo",
                },
                {
                    "event_type": "item_completed",
                    "item_type": "message",
                    "output_index": 0,
                    "identifier": "",
                    "value": "Hello",
                },
            ],
        },
    ]

    async def _delta_iter():
        for delta in deltas:
            yield delta

    events = [
        event
        async for event in iter_response_events(
            _delta_iter(), model_id="test-model", response_id="resp_test"
        )
    ]

    event_types = [event.type for event in events]
    assert event_types[0] == "response.created"
    assert event_types[1] == "response.in_progress"
    assert "response.output_item.added" in event_types
    assert "response.output_text.delta" in event_types
    assert "response.output_text.done" in event_types
    assert event_types[-2] == "response.completed"
    assert event_types[-1] == "done"

    seq_numbers = [
        event.sequence_number
        for event in events
        if hasattr(event, "sequence_number")
    ]
    assert seq_numbers == sorted(seq_numbers)
    assert len(seq_numbers) == len(set(seq_numbers))

    done_text_event = next(
        event for event in events if event.type == "response.output_text.done"
    )
    assert done_text_event.text == "Hello"

    completed_event = next(
        event for event in events if event.type == "response.completed"
    )
    assert completed_event.response.status.value == "completed"
    assert completed_event.response.usage is not None
    assert completed_event.response.usage.input_tokens == 8
    assert completed_event.response.usage.output_tokens == 2
