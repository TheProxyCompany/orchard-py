import asyncio

import pytest

from orchard.server.exceptions import InferenceError
from orchard.server.routes.chat import gather_non_streaming_batch_response


class _IpcStub:
    """Minimal ipc_state stand-in: drains the queue with no liveness checks."""

    async def next_delta(self, queue: asyncio.Queue) -> dict:
        return await queue.get()


@pytest.mark.asyncio
async def test_non_streaming_engine_error_without_indexes_fails_immediately() -> None:
    queue = asyncio.Queue()
    await queue.put(
        {
            "request_id": 1,
            "is_final_delta": True,
            "finish_reason": "error",
            "content": "Engine process disconnected.",
            "error": "Engine process disconnected.",
        }
    )

    with pytest.raises(InferenceError, match="Engine process disconnected"):
        await asyncio.wait_for(
            gather_non_streaming_batch_response(1, queue, _IpcStub(), [1], [1]),
            timeout=0.2,
        )
