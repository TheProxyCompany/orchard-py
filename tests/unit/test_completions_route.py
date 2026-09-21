import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from orchard.app.model_registry import ModelLoadState
from orchard.server.models.completions import CompletionRequest
from orchard.server.routes.completions import handle_completion_request


class _Formatter:
    def __init__(self, *, template_fails: bool = False) -> None:
        self.template_fails = template_fails

    def get_generation_defaults(self, _profile: str) -> dict:
        return {}

    def apply_template(self, _messages: list) -> str:
        if self.template_fails:
            raise ValueError("bad template")
        return "rendered"

    def get_tool_calling_tokens(self) -> dict:
        return {}

    def get_output_frame_tokens(self) -> dict:
        return {}

    def get_thinking_tokens(self) -> dict:
        return {}


class _Registry:
    def __init__(self, formatter: _Formatter) -> None:
        self.info = SimpleNamespace(formatter=formatter, model_path="/models/stub")

    async def schedule_model(self, model_id: str) -> tuple[ModelLoadState, str]:
        return ModelLoadState.READY, model_id

    def get_if_ready(self, _model_id: str) -> SimpleNamespace:
        return self.info


_ERROR_DELTA = {"request_id": 42, "finish_reason": "error", "error": "engine fault"}
_FINAL_DELTA = {
    "request_id": 42,
    "prompt_index": 0,
    "candidate_index": 0,
    "content": "ok",
    "finish_reason": "stop",
    "is_final_delta": True,
}


class _IpcState:
    """Answers the submitted request with one canned delta."""

    response_channel_id = 7

    def __init__(self, delta: dict) -> None:
        self.delta = delta
        self.active_request_queues: dict = {}
        self.cancelled_requests: list[int] = []

    async def get_next_request_id(self) -> int:
        return 42

    async def send_request(self, _request_bytes: bytes) -> None:
        await self.active_request_queues[42].queue.put(dict(self.delta))

    async def next_delta(self, queue: asyncio.Queue) -> dict:
        return await queue.get()

    async def cancel_request(self, request_id: int) -> dict:
        self.cancelled_requests.append(request_id)
        return {"status": "accepted"}


@pytest.mark.asyncio
async def test_gatherer_error_cancels_the_engine_request() -> None:
    # The route used to drop its queue and leave the engine generating every
    # remaining candidate to max tokens for a client that already got a 502.
    ipc_state = _IpcState(_ERROR_DELTA)

    with pytest.raises(HTTPException) as raised:
        await handle_completion_request(
            CompletionRequest(model="stub", prompt="hi", n=2),
            ipc_state,
            _Registry(_Formatter()),
        )

    assert raised.value.status_code == 502
    assert ipc_state.cancelled_requests == [42]
    assert ipc_state.active_request_queues == {}


@pytest.mark.asyncio
async def test_finished_request_is_not_cancelled() -> None:
    ipc_state = _IpcState(_FINAL_DELTA)

    response = await handle_completion_request(
        CompletionRequest(model="stub", prompt="hi"),
        ipc_state,
        _Registry(_Formatter()),
    )

    assert response.choices[0].text == "ok"
    assert ipc_state.cancelled_requests == []
    assert ipc_state.active_request_queues == {}


@pytest.mark.asyncio
async def test_template_error_does_not_leak_the_request_queue() -> None:
    ipc_state = _IpcState(_ERROR_DELTA)

    with pytest.raises(HTTPException) as raised:
        await handle_completion_request(
            CompletionRequest(model="stub", prompt="hi", apply_chat_template=True),
            ipc_state,
            _Registry(_Formatter(template_fails=True)),
        )

    assert raised.value.status_code == 500
    assert ipc_state.active_request_queues == {}
