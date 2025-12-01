import json
from collections import defaultdict

import httpx
import pytest

pytestmark = pytest.mark.asyncio

MODEL_ID = "llama-3.1-8b-instruct"


async def test_chat_completion_multi_candidate_non_streaming(live_server):
    """Verify non-streaming multi-candidate responses return the expected number of choices."""
    server_url = live_server
    candidate_count = 3
    request_payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "user", "content": "Provide three brief facts about the moon."}
        ],
        "max_completion_tokens": 10,
        "temperature": 0.0,
        "stream": False,
        "n": candidate_count,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{server_url}/v1/chat/completions", json=request_payload
        )

    assert response.status_code == 200
    payload = response.json()

    choices = payload.get("choices", [])
    assert len(choices) == candidate_count
    assert {choice["index"] for choice in choices} == set(range(candidate_count))
    for choice in choices:
        message = choice["message"]["content"]
        assert message is not None
        assert message.strip() != ""
        assert choice["finish_reason"]


async def test_chat_completion_multi_candidate_streaming(live_server):
    """Verify streaming multi-candidate responses can be reconstructed per candidate index."""
    server_url = live_server
    candidate_count = 3
    request_payload = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": "Stream three short tips for studying effectively.",
            }
        ],
        "max_completion_tokens": 10,
        "temperature": 0.0,
        "stream": True,
        "n": candidate_count,
    }

    candidate_contents: defaultdict[int, list[str]] = defaultdict(list)
    finish_reasons: dict[int, str] = {}
    saw_done = False

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream(
            "POST", f"{server_url}/v1/chat/completions", json=request_payload
        ) as response:
            assert response.status_code == 200
            async for line in response.aiter_lines():
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    saw_done = True
                    break

                chunk = json.loads(data)
                for choice in chunk.get("choices", []):
                    index = choice["index"]
                    delta = choice.get("delta", {})
                    content_piece = delta.get("content")
                    if content_piece:
                        candidate_contents[index].append(content_piece)

                    finish_reason = choice.get("finish_reason")
                    if finish_reason:
                        finish_reasons[index] = finish_reason

    assert saw_done, "Streaming response did not terminate with [DONE]"
    assert len(candidate_contents) == candidate_count
    assert set(candidate_contents.keys()) == set(range(candidate_count))

    for idx in range(candidate_count):
        # reconstructed = "".join(candidate_contents[idx]).strip()
        # assert reconstructed, f"Candidate {idx} produced no content"
        assert finish_reasons.get(idx), f"Candidate {idx} missing finish reason"
