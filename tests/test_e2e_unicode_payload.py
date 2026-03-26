import json

import httpx
import pytest

pytestmark = pytest.mark.asyncio

async def test_unicode_payload_round_trip(
    live_server, text_model_id, visible_text_completion_floor
):
    target = "😊" * 40  # 40 multi-byte characters (> MAX_INLINE_CONTENT_BYTES)
    replacement_char = "\ufffd"

    request_payload = {
        "model": text_model_id,
        "messages": [
            {"role": "user", "content": "Respond with this emoji: " + target},
        ],
        "temperature": 0.0,
        "max_completion_tokens": max(10, visible_text_completion_floor),
        "stream": True,
    }

    chunks = []
    finish_reason = None

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            f"{live_server}/v1/chat/completions",
            json=request_payload,
        ) as response:
            assert response.status_code == 200

            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue

                payload = line[6:]
                if payload == "[DONE]":
                    break

                chunk = json.loads(payload)
                assert chunk["choices"], (
                    "Expected at least one choice in streamed chunk"
                )

                choice = chunk["choices"][0]
                delta = choice.get("delta", {})
                content = delta.get("content", "")

                if content:
                    assert replacement_char not in content, (
                        "Encountered replacement char in streamed chunk"
                    )
                    chunks.append(content)

                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

    assert chunks, "No streamed content chunks received"
    assert finish_reason is not None, "Expected finish reason in streamed response"

    full_content = "".join(chunks)
    assert full_content, "Expected non-empty streamed content"
    assert replacement_char not in full_content, (
        "Replacement char found in final content"
    )

    total_emojis = full_content.count("😊")
    assert total_emojis > 0, "Expected at least one emoji in the content"
