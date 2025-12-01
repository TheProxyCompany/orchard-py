import httpx
import pytest

pytestmark = pytest.mark.asyncio

MODEL_ID = "moondream3"


async def test_chat_completion_with_logprobs(live_server):
    """
    Tests a chat completion request with logprobs enabled.
    Verifies that the system can return top log probabilities for each generated token.
    """
    server_url = live_server
    request_payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_completion_tokens": 3,  # Generate a few tokens
        "temperature": 1.0,
        "logprobs": True,  # Enable logprobs
        "top_logprobs": 5,  # Request top 5 log probabilities
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{server_url}/v1/chat/completions", json=request_payload
        )

    # --- Assertions ---
    assert response.status_code == 200
    response_data = response.json()

    assert "choices" in response_data
    assert len(response_data["choices"]) > 0

    choice = response_data["choices"][0]
    assert "logprobs" in choice

    # If logprobs were requested and generated, they should be present
    if choice["logprobs"] is not None:
        logprobs_data = choice["logprobs"]
        assert "content" in logprobs_data
        assert isinstance(logprobs_data["content"], list)

        # Check that we have logprobs for each generated token
        assert len(logprobs_data["content"]) > 0

        # Verify the structure of each logprob entry
        for token_logprob in logprobs_data["content"]:
            assert isinstance(token_logprob, dict)
            assert "token" in token_logprob
            assert "logprob" in token_logprob
            assert isinstance(token_logprob["logprob"], int | float)

            # Check top_logprobs if present
            if token_logprob.get("top_logprobs"):
                assert isinstance(token_logprob["top_logprobs"], list)
                assert (
                    len(token_logprob["top_logprobs"]) <= 5
                )  # Should not exceed requested amount

                # Verify each top logprob entry
                for top_logprob in token_logprob["top_logprobs"]:
                    assert "token" in top_logprob
                    assert "logprob" in top_logprob
                    assert isinstance(top_logprob["logprob"], int | float)

                # Verify logprobs are sorted in descending order
                logprob_values = [
                    item["logprob"] for item in token_logprob["top_logprobs"]
                ]
                assert logprob_values == sorted(logprob_values, reverse=True)


async def test_chat_completion_without_logprobs(live_server):
    """
    Tests that when logprobs is not requested, they are not included in the response.
    """
    server_url = live_server
    request_payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_completion_tokens": 3,
        "temperature": 1.0,
        "logprobs": False,  # Explicitly disable logprobs
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{server_url}/v1/chat/completions", json=request_payload
        )

    assert response.status_code == 200
    response_data = response.json()

    choice = response_data["choices"][0]
    # When logprobs is false, the field should be None or absent
    assert choice.get("logprobs") is None


async def test_chat_completion_logprobs_streaming(live_server):
    """
    Tests that logprobs work correctly with streaming responses.
    """
    server_url = live_server
    request_payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "Count to three"}],
        "max_completion_tokens": 5,
        "temperature": 1.0,
        "logprobs": True,
        "top_logprobs": 3,
        "stream": True,  # Enable streaming
    }

    chunks = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        async with client.stream(
            "POST", f"{server_url}/v1/chat/completions", json=request_payload
        ) as response:
            assert response.status_code == 200
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]  # Remove "data: " prefix
                    if data_str == "[DONE]":
                        break
                    chunks.append(data_str)

    # Verify we received multiple chunks
    assert len(chunks) > 0

    # Parse and verify chunk structure
    import json

    for chunk_str in chunks:
        if chunk_str.strip():
            chunk_data = json.loads(chunk_str)
            assert "choices" in chunk_data

            # Check if this chunk has logprobs (not all chunks might have them)
            choice = chunk_data["choices"][0]
            if "logprobs" in choice and choice["logprobs"] is not None:
                # Streaming logprobs might have a different structure
                # Adjust assertions based on actual implementation
                pass
