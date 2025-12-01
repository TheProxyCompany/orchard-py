import httpx
import pytest

pytestmark = pytest.mark.asyncio

MODEL_ID = "moondream3"


async def test_chat_completion_best_of_selects_top_n(live_server):
    """Ensure best_of fan-out returns only the top-n candidates while reflecting total work in usage."""
    server_url = live_server
    best_of = 3
    request_payload = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": "List one fun fact about penguins.",
            }
        ],
        "max_completion_tokens": 8,
        "temperature": 0.2,
        "stream": False,
        "n": 1,
        "best_of": best_of,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{server_url}/v1/chat/completions", json=request_payload
        )

    assert response.status_code == 200
    payload = response.json()

    choices = payload.get("choices", [])
    assert len(choices) == 1, "Only the top-n choice should be returned"
    assert choices[0]["index"] == 0
    assert choices[0]["message"]["content"]

    usage = payload.get("usage", {})
    assert usage, "Usage statistics must be present"
    # Ensure output tokens reflect that all best_of candidates were generated.
    assert usage.get("output_tokens", 0) >= best_of
    assert usage.get("total_tokens", 0) >= usage.get("output_tokens", 0)
    print(choices[0]["message"]["content"])


async def test_chat_completion_best_of_validation_less_than_n(live_server):
    """best_of must be greater than or equal to n."""
    server_url = live_server
    request_payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "Say hello."}],
        "stream": False,
        "n": 2,
        "best_of": 1,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{server_url}/v1/chat/completions", json=request_payload
        )

    assert response.status_code == 422


async def test_chat_completion_best_of_streaming_disallowed(live_server):
    """Streaming responses should reject best_of fan-out."""
    server_url = live_server
    request_payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "Stream a fun fact."}],
        "stream": True,
        "n": 1,
        "best_of": 2,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{server_url}/v1/chat/completions", json=request_payload
        )

    assert response.status_code == 422
