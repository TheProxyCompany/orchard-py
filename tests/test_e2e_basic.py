import httpx
import pytest

pytestmark = pytest.mark.asyncio


async def test_chat_completion_first_token(
    live_server, text_model_id, first_visible_token_budget
):
    """
    Tests a basic, non-streaming chat completion request to the live server.
    Verifies that the system can process a request and return a valid response.
    """
    server_url = live_server
    request_payload = {
        "model": text_model_id,
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_completion_tokens": first_visible_token_budget,
        "temperature": 1.0,  # Use greedy sampling which is implemented
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{server_url}/v1/chat/completions", json=request_payload
        )

    # --- Assertions ---
    assert response.status_code == 200
    response_data = response.json()

    assert "id" in response_data
    assert "choices" in response_data
    assert isinstance(response_data["choices"], list)
    assert len(response_data["choices"]) > 0

    choice = response_data["choices"][0]
    assert "message" in choice
    assert "content" in choice["message"]
    assert choice["message"]["content"] is not None
    assert len(choice["message"]["content"]) > 0

    assert "finish_reason" in choice
    assert choice["finish_reason"].lower() in ["length", "stop"]


async def test_chat_completion_multi_token(live_server, text_model_id):
    """
    Tests a non-streaming request that requires multiple tokens to be generated.
    """
    server_url = live_server
    request_payload = {
        "model": text_model_id,
        "messages": [
            {
                "role": "user",
                "content": "Provide one friendly sentence introducing yourself.",
            }
        ],
        "max_completion_tokens": 64,
        "temperature": 0.0,  # Greedy for deterministic output
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{server_url}/v1/chat/completions", json=request_payload
        )

    assert response.status_code == 200
    response_data = response.json()

    assert len(response_data["choices"]) > 0
    # contents should be deterministic
    content = response_data["choices"][0]["message"]["content"]
    print(content)
    assert content is not None
    assert len(content) > 0
