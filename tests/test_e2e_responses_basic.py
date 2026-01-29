import httpx
import pytest
from helpers import parse_sse_events

pytestmark = pytest.mark.asyncio

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


# ---------------------------------------------------------------------------
# Non-streaming text generation
# ---------------------------------------------------------------------------


async def test_responses_non_streaming_string_input(live_server):
    """String shorthand input produces a valid response."""
    payload = {
        "model": MODEL_ID,
        "input": "Say hello in one sentence.",
        "temperature": 0.0,
        "max_output_tokens": 32,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(f"{live_server}/v1/responses", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert data["object"] == "response"
    assert data["id"].startswith("resp_")
    assert data["status"] == "completed"
    assert data["model"] == MODEL_ID
    assert isinstance(data["created_at"], int)

    assert isinstance(data["output"], list)
    assert len(data["output"]) >= 1
    msg = data["output"][0]
    assert msg["type"] == "message"
    assert msg["role"] == "assistant"
    assert len(msg["content"]) >= 1
    assert msg["content"][0]["type"] == "output_text"
    text = msg["content"][0]["text"]
    assert len(text) > 0
    print(f"Response: {text}")

    usage = data["usage"]
    assert usage["input_tokens"] > 0
    assert usage["output_tokens"] > 0
    assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]


async def test_responses_non_streaming_message_items(live_server):
    """Array-of-items input produces a valid response."""
    payload = {
        "model": MODEL_ID,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": "What is 2+2? Answer with just the number.",
            }
        ],
        "temperature": 0.0,
        "max_output_tokens": 8,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(f"{live_server}/v1/responses", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"

    text = data["output"][0]["content"][0]["text"]
    assert "4" in text
    print(f"Response: {text}")


async def test_responses_echo_fields(live_server):
    """Configuration fields are echoed back in the response."""
    payload = {
        "model": MODEL_ID,
        "input": "Hi",
        "temperature": 0.5,
        "top_p": 0.9,
        "max_output_tokens": 4,
        "metadata": {"test_key": "test_value"},
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(f"{live_server}/v1/responses", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert data["temperature"] == 0.5
    assert data["top_p"] == 0.9
    assert data["metadata"] == {"test_key": "test_value"}


# ---------------------------------------------------------------------------
# Streaming text generation
# ---------------------------------------------------------------------------


async def test_responses_streaming_event_sequence(live_server):
    """Streaming produces the correct SSE event sequence."""
    payload = {
        "model": MODEL_ID,
        "input": "Say hello in one sentence.",
        "temperature": 0.0,
        "max_output_tokens": 32,
        "stream": True,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{live_server}/v1/responses",
            json=payload,
            headers={"Accept": "text/event-stream"},
        )

    assert response.status_code == 200
    events = parse_sse_events(response.text)
    event_types = [e["event"] for e in events]

    assert event_types[0] == "response.created"
    assert event_types[1] == "response.in_progress"
    assert "response.output_item.added" in event_types
    assert "response.content_part.added" in event_types
    assert "response.output_text.delta" in event_types
    assert "response.output_text.done" in event_types
    assert "response.content_part.done" in event_types
    assert "response.output_item.done" in event_types
    assert event_types[-2] == "response.completed"
    assert event_types[-1] == "done"

    seq_numbers = [
        e["data"]["sequence_number"]
        for e in events
        if isinstance(e["data"], dict) and "sequence_number" in e["data"]
    ]
    assert seq_numbers == sorted(seq_numbers)
    assert len(seq_numbers) == len(set(seq_numbers))


async def test_responses_streaming_delta_accumulation(live_server):
    """Accumulated deltas match the done event's full text."""
    payload = {
        "model": MODEL_ID,
        "input": "Count from 1 to 5.",
        "temperature": 0.0,
        "max_output_tokens": 64,
        "stream": True,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{live_server}/v1/responses",
            json=payload,
            headers={"Accept": "text/event-stream"},
        )

    events = parse_sse_events(response.text)

    accumulated = ""
    for e in events:
        if e["event"] == "response.output_text.delta":
            accumulated += e["data"]["delta"]

    done_events = [e for e in events if e["event"] == "response.output_text.done"]
    assert len(done_events) == 1
    assert accumulated == done_events[0]["data"]["text"]
    assert len(accumulated) > 0
    print(f"Streamed: {accumulated}")


async def test_responses_streaming_completed_snapshot(live_server):
    """The response.completed snapshot has correct status and usage."""
    payload = {
        "model": MODEL_ID,
        "input": "Hi",
        "temperature": 0.0,
        "max_output_tokens": 8,
        "stream": True,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{live_server}/v1/responses",
            json=payload,
            headers={"Accept": "text/event-stream"},
        )

    events = parse_sse_events(response.text)

    completed = [e for e in events if e["event"] == "response.completed"]
    assert len(completed) == 1
    snapshot = completed[0]["data"]["response"]
    assert snapshot["status"] == "completed"


# ---------------------------------------------------------------------------
# Incomplete response (max_output_tokens truncation)
# ---------------------------------------------------------------------------


async def test_responses_incomplete_non_streaming(live_server):
    """max_output_tokens=1 results in an incomplete response."""
    payload = {
        "model": MODEL_ID,
        "input": "Write a very long essay about the history of mathematics.",
        "temperature": 0.0,
        "max_output_tokens": 1,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(f"{live_server}/v1/responses", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "incomplete"
    assert data["incomplete_details"]["reason"] == "max_output_tokens"


async def test_responses_incomplete_streaming(live_server):
    """Streaming with max_output_tokens=1 emits response.incomplete."""
    payload = {
        "model": MODEL_ID,
        "input": "Write a very long essay about the history of mathematics.",
        "temperature": 0.0,
        "max_output_tokens": 1,
        "stream": True,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{live_server}/v1/responses",
            json=payload,
            headers={"Accept": "text/event-stream"},
        )

    events = parse_sse_events(response.text)
    event_types = [e["event"] for e in events]

    assert "response.incomplete" in event_types
    incomplete = [e for e in events if e["event"] == "response.incomplete"]
    snapshot = incomplete[0]["data"]["response"]
    assert snapshot["status"] == "incomplete"
    assert snapshot["incomplete_details"]["reason"] == "max_output_tokens"
    assert event_types[-1] == "done"


# ---------------------------------------------------------------------------
# Instructions (system prompt)
# ---------------------------------------------------------------------------


async def test_responses_instructions(live_server):
    """The instructions field works as a system prompt."""
    payload = {
        "model": MODEL_ID,
        "input": "What is your name?",
        "instructions": "You are a helpful assistant named Orchard. Always introduce yourself by name.",
        "temperature": 0.0,
        "max_output_tokens": 64,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(f"{live_server}/v1/responses", json=payload)

    assert response.status_code == 200
    data = response.json()

    text = data["output"][0]["content"][0]["text"].lower()
    assert "orchard" in text
    print(f"Response: {text}")
