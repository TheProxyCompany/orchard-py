import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from orchard.clients.client import Client
from orchard.server.exceptions import InferenceError


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


def _chat_with_terminals(choices):
    client = Client.__new__(Client)
    client._ipc_state = Mock(engine_dead=False)
    queue = asyncio.Queue()
    for index, reason in choices:
        frame = {"request_id": 7, "is_final_delta": True, "finish_reason": reason}
        if index is not None:
            frame.update(prompt_index=0, candidate_index=index)
        if reason == "error":
            frame["error_message"] = "request interrupted"
        else:
            frame["content"] = str(index)
        queue.put_nowait(frame)
    client._open_request = AsyncMock(return_value=(7, queue))
    client._asubmit_request_batch = AsyncMock(return_value=3)
    client._close_request = Mock()
    client._cancel_request_for_cleanup = AsyncMock()
    return client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "choices",
    [
        [(1, "error"), (2, "stop"), (0, "length")],
        [(0, "stop"), (1, "error"), (1, "error"), (2, "length")],
        [(2, "error"), (0, "error"), (1, "error")],
        [(1, "stop"), (None, "error")],
    ],
)
async def test_stream_reports_each_indexed_choice_or_a_whole_request_terminal(choices):
    client = _chat_with_terminals(choices)
    stream = await client.achat("model", [{"role": "user", "content": "tips"}], n=3, stream=True)
    async with asyncio.timeout(1):
        deltas = [delta async for delta in stream]
    assert [(delta.candidate_index, delta.finish_reason) for delta in deltas] == choices
    client._cancel_request_for_cleanup.assert_not_awaited()
    client._close_request.assert_called_once_with(7)


@pytest.mark.asyncio
async def test_non_streaming_chat_still_fails_on_the_first_indexed_error():
    client = _chat_with_terminals([(1, "error")])
    async with asyncio.timeout(1):
        with pytest.raises(InferenceError, match="request interrupted"):
            await client.achat("model", [{"role": "user", "content": "tips"}], n=3, stream=False)
    client._cancel_request_for_cleanup.assert_awaited_once_with(7)
    client._close_request.assert_called_once_with(7)
