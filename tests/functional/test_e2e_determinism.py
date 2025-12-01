import httpx
import pytest

pytestmark = pytest.mark.asyncio

MODEL_ID = "llama-3.1-8b-instruct"

async def test_multi_candidate_determinism(live_server):
    """
    Tests a non-streaming request that requires multiple candidates to be generated.
    """
    server_url = live_server
    request_payload = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": "Provide one friendly sentence introducing yourself.",
            }
        ],
        "max_completion_tokens": 64,
        "temperature": 0.0,  # Greedy for deterministic output
        "stream": False,
        "n": 16,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(f"{server_url}/v1/chat/completions", json=request_payload)

    assert response.status_code == 200
    response_data = response.json()

    assert len(response_data["choices"]) > 0
    # contents should be deterministic
    first_content = response_data["choices"][0]["message"]["content"]
    print(f"First content:\n{first_content}")

    drifts = 0
    for choice in response_data["choices"][1:]:
        assert choice["finish_reason"].lower() in ["length", "stop"]
        assert choice["message"]["content"] is not None
        assert len(choice["message"]["content"]) > 0
        if choice["message"]["content"] != first_content:
            drifts += 1
            print(f"Drift detected!:\n{choice['message']['content']}")

    assert drifts == 0, f"Expected 0 drifts, got {drifts}"


async def test_sequential_request_determinism(live_server):
    """
    Tests a sequential request that generates the same content multiple times.
    """
    server_url = live_server
    request_payload = {
        "model": MODEL_ID,
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

    async def make_request():
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{server_url}/v1/chat/completions", json=request_payload)
        return response

    first_response = None
    valid_responses = 0
    num_requests = 3
    for i in range(num_requests):
        response = await make_request()
        assert response.status_code == 200
        response_data = response.json()

        assert len(response_data["choices"]) > 0
        # contents should be deterministic
        content = response_data["choices"][0]["message"]["content"]
        if first_response is None:
            first_response = content
            valid_responses += 1
        elif first_response == content:
            valid_responses += 1
        else:
            print(f"Drift detected!:\n{content} at index {i}")
            break

    print(first_response)
    assert valid_responses == num_requests, f"Expected {num_requests} valid responses, got {valid_responses}"
