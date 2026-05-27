import asyncio

import pytest

from orchard.server.exceptions import InferenceError
from orchard.server.routes.chat import gather_non_streaming_batch_response


@pytest.mark.asyncio
async def test_non_streaming_engine_disconnect_without_indexes_fails_immediately() -> None:
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
            gather_non_streaming_batch_response(1, queue, [1], [1]),
            timeout=0.2,
        )
