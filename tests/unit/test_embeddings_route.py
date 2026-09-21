import asyncio
import struct
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from orchard.app.model_registry import ModelLoadState
from orchard.server.models.embeddings import EmbeddingRequest
from orchard.server.routes.embeddings import create_embeddings


class _Registry:
    async def schedule_model(self, model_id: str) -> tuple[ModelLoadState, str]:
        return ModelLoadState.READY, model_id

    def get_if_ready(self, _model_id: str) -> SimpleNamespace:
        return SimpleNamespace(model_path="/models/stub")


class _IpcState:
    """Answers the submitted request with `delta`, or never."""

    response_channel_id = 7
    lossless_responses = True

    def __init__(self, delta: dict | None) -> None:
        self.delta = delta
        self.active_request_queues: dict = {}
        self.cancelled_requests: list[int] = []

    async def get_next_request_id(self) -> int:
        return 42

    async def send_request(self, _request_bytes: bytes) -> None:
        if self.delta is not None:
            await self.active_request_queues[42].queue.put(dict(self.delta))

    async def next_delta(self, queue: asyncio.Queue) -> dict:
        if self.delta is None:
            raise TimeoutError
        return await queue.get()

    async def cancel_request(self, request_id: int) -> dict:
        self.cancelled_requests.append(request_id)
        return {"status": "accepted"}


@pytest.mark.asyncio
async def test_a_request_that_times_out_is_cancelled_in_the_engine() -> None:
    # The route used to forget the request and leave it queued or prefilling
    # in the engine for a caller that already got a 502.
    ipc_state = _IpcState(None)

    with pytest.raises(HTTPException) as raised:
        await create_embeddings(
            EmbeddingRequest(model="stub", input="hi"), ipc_state, _Registry()
        )

    assert raised.value.status_code == 502
    assert ipc_state.cancelled_requests == [42]
    assert ipc_state.active_request_queues == {}


@pytest.mark.asyncio
async def test_a_finished_request_is_not_cancelled() -> None:
    ipc_state = _IpcState(
        {
            "request_id": 42,
            "prompt_token_count": 1,
            "embedding_bytes": struct.pack("2f", 0.5, -1.0),
            "is_final_delta": True,
        }
    )

    response = await create_embeddings(
        EmbeddingRequest(model="stub", input="hi"), ipc_state, _Registry()
    )

    assert response.data[0].embedding == [0.5, -1.0]
    assert ipc_state.cancelled_requests == []
    assert ipc_state.active_request_queues == {}
