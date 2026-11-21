import asyncio
import struct
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from starlette.requests import Request

from orchard.server.exceptions import InferenceError
from orchard.server.models.embeddings import EmbeddingRequest
from orchard.server.routes import embeddings
from orchard.server.routes.embeddings import gather_embedding_response


@pytest.mark.asyncio
async def test_embedding_results_follow_input_order_and_count_usage_once():
    queue = asyncio.Queue()
    for delta in [
        {"prompt_index": 1, "prompt_token_count": 3, "embedding_bytes": struct.pack("2f", 5, 8)},
        {"prompt_index": 1, "prompt_token_count": 3, "is_final_delta": True},
        {"prompt_index": 1, "is_final_delta": True},
        {"prompt_index": 0, "prompt_token_count": 2, "embedding_bytes": struct.pack("2f", 1, 2), "is_final_delta": True},
    ]:
        queue.put_nowait(delta)
    response = await gather_embedding_response(42, queue, SimpleNamespace(next_delta=lambda q: q.get()), 2)
    assert response["embedding_vectors"] == [[1, 2], [5, 8]]
    assert response["prompt_tokens"] == 5
    await queue.join()


@pytest.mark.asyncio
@pytest.mark.parametrize("index", [None, -1, 2])
async def test_embedding_batch_rejects_unidentified_results(index):
    queue = asyncio.Queue()
    queue.put_nowait({"prompt_index": index, "is_final_delta": True})
    with pytest.raises(InferenceError, match="Invalid embedding prompt index"):
        await gather_embedding_response(42, queue, SimpleNamespace(next_delta=lambda q: q.get()), 2)
    await queue.join()


@pytest.mark.asyncio
async def test_disconnected_embedding_client_cancels_queued_inference(monkeypatch):
    sent = asyncio.Event()

    async def submit(_payload):
        sent.set()

    async def receive():
        await sent.wait()
        return {"type": "http.disconnect"}

    ipc = SimpleNamespace(
        get_next_request_id=AsyncMock(return_value=42),
        send_request=submit,
        next_delta=lambda queue: queue.get(),
        cancel_request=AsyncMock(),
        response_channel_id=7,
        lossless_responses=True,
        active_request_queues={},
    )
    model = SimpleNamespace(model_path="/fixture")
    monkeypatch.setattr(embeddings, "resolve_model", AsyncMock(return_value=("model", model)))
    result = await asyncio.wait_for(
        embeddings.create_embeddings(
            request=EmbeddingRequest(model="model", input=["first", "second"]),
            http_request=Request({"type": "http"}, receive),
            ipc_state=ipc,
            model_registry=None,
            model_directories=[],
        ),
        timeout=2,
    )
    assert result.status_code == 499
    ipc.cancel_request.assert_awaited_once_with(42)
    assert not ipc.active_request_queues
