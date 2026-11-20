import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from orchard.clients.client import Client


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False])
async def test_chat_waits_for_every_prompt_and_candidate(stream):
    client = Client.__new__(Client)
    client._ipc_state = Mock(engine_dead=False)
    queue = asyncio.Queue()
    for prompt, candidate in [(1, 0), (0, 1), (0, 0), (1, 1)]:
        queue.put_nowait(
            {
                "request_id": 7,
                "prompt_index": prompt,
                "candidate_index": candidate,
                "is_final_delta": True,
                "finish_reason": "length",
                "tokens": [32 + candidate],
                "content": "x",
            }
        )
    client._open_request = AsyncMock(return_value=(7, queue))
    client._asubmit_request_batch = AsyncMock(return_value=4)
    client._close_request = Mock()
    client._cancel_request_for_cleanup = AsyncMock()
    messages = [[{"role": "user", "content": str(i)}] for i in range(2)]
    result = await client.achat("model", messages, n=2, stream=stream)
    if stream:
        deltas = [delta async for delta in result]
    else:
        deltas = [delta for response in result for delta in response.deltas]

    assert {(d.prompt_index, d.candidate_index) for d in deltas if d.is_final} == {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    }
    client._close_request.assert_called_once_with(7)
    client._cancel_request_for_cleanup.assert_not_awaited()


@pytest.mark.asyncio
async def test_chat_exits_if_a_parent_fails_before_creating_its_candidates():
    client = Client.__new__(Client)
    queue = asyncio.Queue()
    queue.put_nowait(
        {
            "request_id": 8,
            "prompt_index": 0,
            "candidate_index": 0,
            "is_final_delta": True,
            "finish_reason": "error",
            "error_message": "could not run the parent",
        }
    )
    cancel = AsyncMock()
    deltas = [
        delta
        async for delta in client._async_process_stream(
            queue, on_cancel=cancel, expected_final_count=16
        )
    ]
    assert [delta.error_message for delta in deltas] == ["could not run the parent"]
    cancel.assert_awaited_once()
