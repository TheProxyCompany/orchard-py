import httpx
import pytest

from tests.functional.cases._timeout import HTTP_TIMEOUT_S

pytestmark = pytest.mark.asyncio


async def test_chat_completion_respects_stop_sequence(
    live_server, engine, text_model_id
):
    server_url = live_server
    payload = {
        "model": text_model_id,
        "messages": [
            {
                "role": "user",
                "content": "Reply with exactly: red, white, blue",
            }
        ],
        "temperature": 0.0,
        "reasoning": False,
        "stream": False,
        "max_completion_tokens": 32,
        "stop": ["blue"],
        "logprobs": True,
        "top_logprobs": 10,
    }

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
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
    model_info = engine.model_registry().get_if_ready(text_model_id)
    assert model_info is not None
    if model_info.releases_held_text:
        # The stop sequence ends the reply and is not part of it: the engine's
        # text stream leaves it out, and only the decoded tokens spell it.
        assert "blue" not in normalized
    else:
        # Against an engine without released_text the route still reads the
        # decoded tokens where it did, and they spell the stop sequence.
        assert normalized.endswith("blue")

    assert choice.get("finish_reason", "").lower() == "stop"
    print(content)
