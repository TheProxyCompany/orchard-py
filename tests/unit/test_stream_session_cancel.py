import asyncio

import pytest

from orchard.ipc.utils import ResponseDeltaDict
from orchard.server.routes._common import managed_stream_session


class _FakeIPCState:
    def __init__(self) -> None:
        self.active_request_queues = {}
        self.cancelled_requests: list[int] = []

    async def cancel_request(self, request_id: int) -> dict:
        self.cancelled_requests.append(request_id)
        return {"status": "accepted"}


@pytest.mark.asyncio
async def test_managed_stream_session_cancels_unfinished_request_on_exit() -> None:
    ipc_state = _FakeIPCState()
    queue: asyncio.Queue[ResponseDeltaDict] = asyncio.Queue()

    async with managed_stream_session(
        ipc_state,
        42,
        queue,
        cancel_on_exit=True,
        completed=lambda: False,
    ):
        assert 42 in ipc_state.active_request_queues

    assert 42 not in ipc_state.active_request_queues
    assert ipc_state.cancelled_requests == [42]


@pytest.mark.asyncio
async def test_managed_stream_session_does_not_cancel_completed_request() -> None:
    ipc_state = _FakeIPCState()
    queue: asyncio.Queue[ResponseDeltaDict] = asyncio.Queue()

    async with managed_stream_session(
        ipc_state,
        42,
        queue,
        cancel_on_exit=True,
        completed=lambda: True,
    ):
        assert 42 in ipc_state.active_request_queues

    assert ipc_state.cancelled_requests == []
