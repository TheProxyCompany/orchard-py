from __future__ import annotations

import json
from collections.abc import Iterator

import pytest

from orchard.clients import Client
from orchard.server.models.responses.output import ResponseObject
from orchard.server.models.responses.tools import Function

pytestmark = pytest.mark.asyncio

MODEL_IDS = ["meta-llama/Llama-3.1-8B-Instruct", "moondream3"]
TOOL_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

WEATHER_TOOL = Function(
    name="get_weather",
    description="Get the current weather for a location.",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name, e.g. 'San Francisco'",
            }
        },
        "required": ["location"],
    },
)


@pytest.mark.parametrize("model_id", MODEL_IDS)
async def test_client_responses_non_streaming_text(
    client: Client, model_id: str
) -> None:
    result = await client.aresponses(
        model_id,
        input="Say hello in one sentence.",
        temperature=0.0,
        max_output_tokens=32,
    )

    assert isinstance(result, ResponseObject)
    assert result.status.value == "completed"
    assert result.model == model_id
    assert result.output_text.strip()
    assert result.usage is not None
    assert result.usage.input_tokens > 0
    assert result.usage.output_tokens > 0


@pytest.mark.parametrize("model_id", MODEL_IDS)
async def test_client_responses_non_streaming_message_items(
    client: Client, model_id: str
) -> None:
    result = await client.aresponses(
        model_id,
        input=[
            {
                "type": "message",
                "role": "user",
                "content": "What is 2+2? Answer with just the number.",
            }
        ],
        temperature=0.0,
        max_output_tokens=8,
    )
    assert isinstance(result, ResponseObject)
    assert result.status.value == "completed"
    assert "4" in result.output_text


@pytest.mark.parametrize("model_id", MODEL_IDS)
async def test_client_responses_streaming_text(
    client: Client, model_id: str
) -> None:
    stream = await client.aresponses(
        model_id,
        input="Count from 1 to 5.",
        stream=True,
        temperature=0.0,
        max_output_tokens=64,
    )
    assert hasattr(stream, "__aiter__")

    events = [event async for event in stream]
    event_types = [event.type for event in events]

    assert event_types[0] == "response.created"
    assert event_types[1] == "response.in_progress"
    assert "response.output_item.added" in event_types
    assert "response.output_text.delta" in event_types
    assert "response.output_text.done" in event_types
    assert "response.output_item.done" in event_types
    assert event_types[-2] in {"response.completed", "response.incomplete"}
    assert event_types[-1] == "done"

    accumulated = "".join(
        event.delta for event in events if event.type == "response.output_text.delta"
    )
    done_events = [event for event in events if event.type == "response.output_text.done"]
    assert done_events
    assert accumulated == done_events[0].text
    assert accumulated.strip()


@pytest.mark.parametrize("model_id", MODEL_IDS)
async def test_client_responses_streaming_incomplete(
    client: Client, model_id: str
) -> None:
    stream = await client.aresponses(
        model_id,
        input="Write a very long essay about the history of mathematics.",
        stream=True,
        temperature=0.0,
        max_output_tokens=1,
    )
    events = [event async for event in stream]
    event_types = [event.type for event in events]

    assert "response.incomplete" in event_types
    incomplete = next(event for event in events if event.type == "response.incomplete")
    assert incomplete.response.status.value == "incomplete"
    assert incomplete.response.incomplete_details is not None
    assert incomplete.response.incomplete_details.reason == "max_output_tokens"
    assert event_types[-1] == "done"


async def test_client_responses_tool_calling(client: Client) -> None:
    result = await client.aresponses(
        TOOL_MODEL_ID,
        input=[
            {
                "type": "message",
                "role": "system",
                "content": "You are a helpful assistant with tool calling capabilities. "
                "When you receive a tool call response, use the output to format an answer.",
            },
            {
                "type": "message",
                "role": "user",
                "content": "What's the weather in San Francisco?",
            },
        ],
        tools=[WEATHER_TOOL],
        tool_choice="required",
        temperature=0.0,
        max_output_tokens=128,
    )
    assert isinstance(result, ResponseObject)
    assert result.status.value == "completed"

    assert result.tool_calls, f"Expected function_call output item, got: {result.output}"
    call = result.tool_calls[0]
    assert call.name == "get_weather"
    assert call.call_id
    assert call.status.value == "completed"
    parsed_args = json.loads(call.arguments)
    assert isinstance(parsed_args, dict)
    assert "location" in parsed_args


async def test_client_responses_tool_result_continuation(client: Client) -> None:
    result_1 = await client.aresponses(
        TOOL_MODEL_ID,
        input=[
            {
                "type": "message",
                "role": "system",
                "content": "You are a helpful assistant with tool calling capabilities. "
                "When you receive a tool call response, use the output to format an answer.",
            },
            {
                "type": "message",
                "role": "user",
                "content": "What's the weather in San Francisco?",
            },
        ],
        tools=[WEATHER_TOOL],
        tool_choice="required",
        temperature=0.0,
        max_output_tokens=128,
    )
    assert isinstance(result_1, ResponseObject)

    assert result_1.tool_calls
    tool_call = result_1.tool_calls[0]

    result_2 = await client.aresponses(
        TOOL_MODEL_ID,
        input=[
            {
                "type": "message",
                "role": "system",
                "content": "You are a helpful assistant with tool calling capabilities. "
                "When you receive a tool call response, use the output to format an answer.",
            },
            {
                "type": "message",
                "role": "user",
                "content": "What's the weather in San Francisco?",
            },
            {
                "type": "function_call",
                "call_id": tool_call.call_id,
                "name": tool_call.name,
                "arguments": tool_call.arguments,
            },
            {
                "type": "function_call_output",
                "call_id": tool_call.call_id,
                "output": json.dumps(
                    {"temperature": 65, "unit": "fahrenheit", "condition": "foggy"}
                ),
            },
        ],
        tools=[WEATHER_TOOL],
        temperature=0.0,
        max_output_tokens=128,
    )
    assert isinstance(result_2, ResponseObject)
    assert result_2.status.value == "completed"
    text = result_2.output_text.lower()
    assert any(token in text for token in ("65", "fog", "san francisco"))


@pytest.mark.parametrize("model_id", MODEL_IDS)
async def test_client_responses_instructions(client: Client, model_id: str) -> None:
    result = await client.aresponses(
        model_id,
        input="What is your name?",
        instructions="You are a helpful assistant named Orchard. Always introduce yourself by name.",
        temperature=0.0,
        max_output_tokens=64,
    )
    assert isinstance(result, ResponseObject)
    assert result.status.value == "completed"
    text = result.output_text.lower()
    if model_id == TOOL_MODEL_ID:
        assert "orchard" in text


@pytest.mark.parametrize("model_id", MODEL_IDS)
async def test_client_responses_sync_wrapper(client: Client, model_id: str) -> None:
    non_streaming = client.responses(
        model_id,
        input="Say hello in one short sentence.",
        temperature=0.0,
        max_output_tokens=32,
    )
    assert isinstance(non_streaming, ResponseObject)
    assert non_streaming.status.value == "completed"
    assert non_streaming.output_text

    streaming = client.responses(
        model_id,
        input="Count from 1 to 3.",
        stream=True,
        temperature=0.0,
        max_output_tokens=32,
    )
    assert isinstance(streaming, Iterator)

    events = list(streaming)
    assert events
    assert events[0].type == "response.created"
    assert events[-1].type == "done"


@pytest.mark.parametrize("model_id", MODEL_IDS)
async def test_client_responses_text_helpers(client: Client, model_id: str) -> None:
    async_chunks = [
        chunk
        async for chunk in client.aresponses_text(
            model_id,
            input="Write one short sentence about the sky.",
            temperature=0.0,
            max_output_tokens=32,
        )
    ]
    assert "".join(async_chunks).strip()

    sync_chunks = list(
        client.responses_text(
            model_id,
            input="Write one short sentence about the ocean.",
            temperature=0.0,
            max_output_tokens=32,
        )
    )
    assert "".join(sync_chunks).strip()
