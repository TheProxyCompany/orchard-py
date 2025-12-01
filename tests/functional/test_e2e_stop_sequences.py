import httpx
import pytest

pytestmark = pytest.mark.asyncio

MODEL_ID = "moondream3"


async def test_chat_completion_respects_stop_sequence(live_server):
    server_url = live_server
    payload = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": "What are the national colors of the United States of America?",
            }
        ],
        "temperature": 0.0,
        "stream": False,
        "stop": ["blue"],
        "logprobs": True,
        "top_logprobs": 10,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{server_url}/v1/chat/completions",
            json=payload,
        )

    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert data["choices"]

    choice = data["choices"][0]
    content = choice["message"]["content"] or ""

    normalized = content.lower()
    assert "red" in normalized
    assert "white" in normalized
    assert "blue" in normalized

    assert normalized.endswith("blue")

    assert choice.get("finish_reason", "").lower() == "stop"
    print(content)
