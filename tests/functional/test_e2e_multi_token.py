import json

import httpx
import pytest

pytestmark = pytest.mark.asyncio

MODEL_ID = "llama-3.1-8b-instruct"


async def test_chat_completion_multi_token_non_streaming(live_server):
    """
    Tests that the engine can correctly answer a simple question,
    verifying the RoPE offset logic is fixed.
    """
    server_url = live_server
    request_payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "temperature": 0.0,  # Use greedy sampling for a deterministic answer
        "max_completion_tokens": 10,
        "logprobs": True,
        "top_logprobs": 5,
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{server_url}/v1/chat/completions", json=request_payload
        )

    # --- Assertions ---
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
    response_data = response.json()

    assert "id" in response_data
    assert "choices" in response_data
    assert isinstance(response_data["choices"], list)
    assert len(response_data["choices"]) > 0

    choice = response_data["choices"][0]
    assert "message" in choice
    if "content" in choice["message"] and choice["message"]["content"] is not None:
        # The model's greedy output should be "The capital of France is Paris." or something very similar.
        # We check if "Paris" is in the generated text.
        if "Paris" not in choice["message"]["content"]:
            # Pretty print logprobs for the second token
            if choice.get("logprobs"):
                logprobs_data = choice["logprobs"]
                if logprobs_data.get("content"):
                    for i, data in enumerate(logprobs_data["content"]):
                        print(f"Token {i}: {data.get('token', 'N/A')}")
                        print(f"Logprob {i}: {data.get('logprob', 'N/A')}")
                        if "top_logprobs" in data:
                            print("\nTop Logprobs:")
                            for entry in data["top_logprobs"][:5]:
                                print(
                                    f"  {entry.get('token', 'N/A'):20s} -> {entry.get('logprob', 'N/A'):.4f}"
                                )
                    print("=" * 30 + "\n")
        assert (
            "Paris" in choice["message"]["content"]
        ), f"Expected 'Paris' in response but got: '{choice['message']['content']}'"
        print(choice["message"]["content"])
    else:
        pytest.fail(f"Expected 'content' in choice['message'], but got: {choice}")

    # Assert that the generation was stopped because of the token limit
    assert choice["finish_reason"].lower() in ["stop", "length"]
    assert "usage" in response_data, "Expected 'usage' field in response"
    usage = response_data["usage"]

    assert (
        usage.get("input_tokens", 0) > 0
    ), "Expected 'input_tokens' to be greater than 0"
    assert usage.get("total_tokens") == usage.get("input_tokens", 0) + usage.get(
        "output_tokens", 0
    )


async def test_chat_completion_multi_token_streaming(live_server):
    """
    Tests a multi-token, streaming chat completion request.
    Verifies that the system streams multiple tokens correctly.
    """
    server_url = live_server
    request_payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "user", "content": "Tell me a very short story in one sentence."}
        ],
        "max_completion_tokens": 10,
        "temperature": 0.0,
        "stream": True,
    }

    chunks_received = 0
    full_content = ""
    finish_reason = None

    async with httpx.AsyncClient(timeout=30.0) as client:
        async with client.stream(
            "POST",
            f"{server_url}/v1/chat/completions",
            json=request_payload,
        ) as response:
            assert response.status_code == 200

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]  # Remove "data: " prefix
                    if data_str == "[DONE]":
                        break

                    chunk_data = json.loads(data_str)
                    chunks_received += 1

                    assert "id" in chunk_data
                    assert "choices" in chunk_data
                    assert len(chunk_data["choices"]) > 0

                    choice = chunk_data["choices"][0]
                    if "delta" in choice and "content" in choice["delta"]:
                        content = choice["delta"]["content"]
                        if content:
                            full_content += content

                    if "finish_reason" in choice:
                        finish_reason = choice["finish_reason"]

    # --- Assertions ---
    assert chunks_received > 1, f"Expected multiple chunks, but got {chunks_received}"
    assert len(full_content) > 0, "Expected non-empty content"
    assert finish_reason is not None, "Expected a finish reason"
    assert finish_reason.lower() in ["stop", "length"]
