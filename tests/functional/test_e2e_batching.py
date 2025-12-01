import httpx
import pytest

pytestmark = pytest.mark.asyncio

MODEL_ID = "moondream3"


async def test_chat_completion_batched_homogeneous(live_server):
    server_url = live_server
    request_payload = {
        "model": MODEL_ID,
        "messages": [
            [{"role": "user", "content": "Say hello politely."}],
            [{"role": "user", "content": "Give me a fun fact about space."}],
        ],
        "max_completion_tokens": 10,
        "temperature": 0.0,
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{server_url}/v1/chat/completions", json=request_payload
        )

    assert response.status_code == 200
    data = response.json()

    assert "choices" in data
    choices = data["choices"]
    assert isinstance(choices, list)
    assert len(choices) == 2

    for index, choice in enumerate(choices):
        assert choice["index"] == index
        message = choice["message"]
        print(message["content"])
        assert message["role"] == "assistant"
        assert message["content"]
        assert choice["finish_reason"] is not None

    usage = data.get("usage", {})
    assert usage.get("total_tokens") is not None


async def test_chat_completion_batched_heterogeneous(live_server):
    server_url = live_server
    request_payload = {
        "model": MODEL_ID,
        "messages": [
            [{"role": "user", "content": "Respond with a single word greeting."}],
            [{"role": "user", "content": "List three colors separated by commas."}],
        ],
        "max_completion_tokens": [1, 4],
        "temperature": [0.0, 0.0],
        "stop": [["!"], [","]],
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{server_url}/v1/chat/completions", json=request_payload
        )

    assert response.status_code == 200
    data = response.json()
    choices = data["choices"]
    assert len(choices) == 2

    first_choice, second_choice = choices

    assert first_choice["finish_reason"] is not None
    assert second_choice["finish_reason"] is not None


async def test_chat_completion_batch_length_mismatch_returns_422(live_server):
    server_url = live_server
    request_payload = {
        "model": MODEL_ID,
        "messages": [
            [{"role": "user", "content": "Prompt one"}],
            [{"role": "user", "content": "Prompt two"}],
        ],
        "temperature": [0.2, 0.4, 0.6],
        "max_completion_tokens": 2,
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{server_url}/v1/chat/completions", json=request_payload
        )

    assert response.status_code == 422
