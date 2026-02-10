import json

import httpx
import pytest
from helpers import parse_sse_events

pytestmark = pytest.mark.asyncio

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

WEATHER_TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": "Get the current weather for a location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name, e.g. 'San Francisco'",
            },
        },
        "required": ["location"],
    },
}


# ---------------------------------------------------------------------------
# Tool calling (non-streaming)
# ---------------------------------------------------------------------------


async def test_responses_tool_call_non_streaming(live_server):
    """Model emits a function call when given tools and a triggering prompt."""
    payload = {
        "model": MODEL_ID,
        "input": [
            {
                "type": "message",
                "role": "system",
                "content": "You are a helpful assistant with tool calling capabilities. When you receive a tool call response, use the output to format an answer to the orginal user question.",
            },
            {
                "type": "message",
                "role": "user",
                "content": "What's the weather in San Francisco?",
            },
        ],
        "tools": [WEATHER_TOOL],
        "tool_choice": "required",
        "temperature": 0.0,
        "max_output_tokens": 128,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(f"{live_server}/v1/responses", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"

    tool_calls = [item for item in data["output"] if item["type"] == "function_call"]
    assert len(tool_calls) >= 1, f"Expected function_call in output: {data['output']}"

    call = tool_calls[0]
    assert call["name"] == "get_weather"
    assert call["call_id"]
    assert call["status"] == "completed"

    args = json.loads(call["arguments"])
    assert isinstance(args, dict)
    assert "location" in args
    print(f"Tool call: {call['name']}({call['arguments']})")


# ---------------------------------------------------------------------------
# Tool calling (streaming)
# ---------------------------------------------------------------------------


async def test_responses_tool_call_streaming(live_server):
    """Streaming tool call produces the correct SSE event sequence."""
    payload = {
        "model": MODEL_ID,
        "input": [
            {
                "type": "message",
                "role": "system",
                "content": "You are a helpful assistant with tool calling capabilities. When you receive a tool call response, use the output to format an answer to the orginal user question.",
            },
            {
                "type": "message",
                "role": "user",
                "content": "What's the weather in San Francisco?",
            },
        ],
        "tools": [WEATHER_TOOL],
        "tool_choice": "required",
        "temperature": 0.0,
        "max_output_tokens": 128,
        "stream": True,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{live_server}/v1/responses",
            json=payload,
            headers={"Accept": "text/event-stream"},
        )

    assert response.status_code == 200
    events = parse_sse_events(response.text)
    event_types = [e["event"] for e in events]

    assert "response.function_call_arguments.delta" in event_types, (
        f"Missing function_call_arguments.delta in: {event_types}"
    )
    assert "response.function_call_arguments.done" in event_types

    accumulated_args = ""
    for e in events:
        if e["event"] == "response.function_call_arguments.delta":
            accumulated_args += e["data"]["delta"]

    done_events = [
        e for e in events if e["event"] == "response.function_call_arguments.done"
    ]
    assert len(done_events) >= 1
    assert accumulated_args == done_events[0]["data"]["arguments"]

    parsed = json.loads(done_events[0]["data"]["arguments"])
    assert isinstance(parsed, dict)
    print(f"Streamed tool call args: {done_events[0]['data']['arguments']}")

    item_done_events = [e for e in events if e["event"] == "response.output_item.done"]
    fc_items = [
        e for e in item_done_events if e["data"]["item"]["type"] == "function_call"
    ]
    assert len(fc_items) >= 1
    assert fc_items[0]["data"]["item"]["name"] == "get_weather"

    assert event_types[-1] == "done"


# ---------------------------------------------------------------------------
# Tool result continuation (multi-turn)
# ---------------------------------------------------------------------------


async def test_responses_tool_result_continuation(live_server):
    """Client sends tool result back and model continues with a message."""
    # Turn 1: trigger tool call
    payload_1 = {
        "model": MODEL_ID,
        "input": [
            {
                "type": "message",
                "role": "system",
                "content": "You are a helpful assistant with tool calling capabilities. When you receive a tool call response, use the output to format an answer to the orginal user question.",
            },
            {
                "type": "message",
                "role": "user",
                "content": "What's the weather in San Francisco?",
            },
        ],
        "tools": [WEATHER_TOOL],
        "tool_choice": "required",
        "temperature": 0.0,
        "max_output_tokens": 128,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp1 = await client.post(f"{live_server}/v1/responses", json=payload_1)

    assert resp1.status_code == 200
    data1 = resp1.json()

    tool_calls = [item for item in data1["output"] if item["type"] == "function_call"]
    assert len(tool_calls) >= 1
    call = tool_calls[0]

    # Turn 2: send back the tool result
    payload_2 = {
        "model": MODEL_ID,
        "input": [
            {
                "type": "message",
                "role": "system",
                "content": "You are a helpful assistant with tool calling capabilities. When you receive a tool call response, use the output to format an answer to the orginal user question.",
            },
            {
                "type": "message",
                "role": "user",
                "content": "What's the weather in San Francisco?",
            },
            {
                "type": "function_call",
                "call_id": call["call_id"],
                "name": call["name"],
                "arguments": call["arguments"],
            },
            {
                "type": "function_call_output",
                "call_id": call["call_id"],
                "output": json.dumps(
                    {"temperature": 65, "unit": "fahrenheit", "condition": "foggy"}
                ),
            },
        ],
        "tools": [WEATHER_TOOL],
        "temperature": 0.0,
        "max_output_tokens": 128,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp2 = await client.post(f"{live_server}/v1/responses", json=payload_2)

    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["status"] == "completed"

    messages = [item for item in data2["output"] if item["type"] == "message"]
    assert len(messages) >= 1

    text = messages[0]["content"][0]["text"].lower()
    assert any(word in text for word in ["65", "fog", "san francisco"]), (
        f"Model didn't incorporate tool result: {text}"
    )
    print(f"Continuation: {text}")
