import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import anyio
import pytest

from orchard.server.routes._common import managed_stream_session


@pytest.mark.asyncio
async def test_disconnected_http_stream_cancels_the_request_and_releases_queued_data():
    acknowledged = []

    async def cancel_request(request_id):
        await anyio.lowlevel.checkpoint()
        acknowledged.append(request_id)

    state = SimpleNamespace(active_request_queues={}, cancel_request=cancel_request)
    queue = asyncio.Queue()
    buffer = memoryview(bytearray(b"pending response"))
    queue.put_nowait({"bulk_content_view": buffer})

    with anyio.CancelScope() as connection:
        async with managed_stream_session(state, 17, queue, cancel_on_exit=True):
            connection.cancel()
            await anyio.lowlevel.checkpoint()

    assert acknowledged == [17]
    assert not state.active_request_queues
    assert queue.empty()
    with pytest.raises(ValueError):
        buffer.tobytes()


@pytest.mark.asyncio
async def test_request_queue_is_released_when_native_cancellation_is_interrupted():
    state = SimpleNamespace(
        active_request_queues={},
        cancel_request=AsyncMock(side_effect=asyncio.CancelledError()),
    )
    queue = asyncio.Queue()
    queue.put_nowait({"content": "pending response"})
    with pytest.raises(asyncio.CancelledError):
        async with managed_stream_session(state, 29, queue, cancel_on_exit=True):
            pass

    assert not state.active_request_queues
    assert queue.empty()
