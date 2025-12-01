from __future__ import annotations

from collections.abc import Iterator

import pytest

from proxy_inference_engine.clients import Client
from proxy_inference_engine.engine import ClientDelta, ClientResponse

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("model_id", ["llama-3.1-8b-instruct", "moondream3"])
@pytest.mark.parametrize(
    "prompt",
    [
        "You have 5 output tokens. Respond with a 5 token poem.",
        "You have 5 output tokens. Respond with a 5 token plea for more tokens.",
    ],
)
async def test_client_chat_non_streaming(
    client: Client,
    model_id: str,
    prompt: str,
) -> None:
    response = await client.achat(
        model_id,
        [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        stream=False,
        temperature=0.0,
        max_generated_tokens=5,
    )
    print(f"User: {prompt}")
    assert isinstance(response, ClientResponse)
    assert response.text.strip()
    print(f"{model_id}: {response.text}")
    assert response.usage.completion_tokens > 0
    assert response.usage.completion_tokens == 5


@pytest.mark.parametrize("model_id", ["llama-3.1-8b-instruct", "moondream3"])
@pytest.mark.parametrize(
    "prompt",
    [
        "Respond with your favorite musical artist of the last 10 years.",
        "Respond with your favorite movie of the last 10 years.",
    ],
)
async def test_client_chat_streaming(
    client: Client, model_id: str, prompt: str
) -> None:
    stream = client.chat(
        model_id,
        [{"role": "user", "content": prompt}],
        stream=True,
        temperature=0.7,
        max_generated_tokens=96,
    )
    print(f"User: {prompt}")
    print(f"{model_id}: ", end="", flush=True)
    assert isinstance(stream, Iterator)
    deltas: list[ClientDelta] = []
    content = ""
    for delta in stream:
        deltas.append(delta)
        print(delta.content, end="", flush=True)
        content += delta.content or ""
    print()
    assert len(deltas) > 1
    assert content.strip()
