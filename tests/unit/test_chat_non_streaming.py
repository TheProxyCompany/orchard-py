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


def test_every_route_calls_the_gatherer_with_its_full_signature() -> None:
    # /v1/completions kept calling it without ipc_state after the parameter was
    # added, so every non-streaming completion raised TypeError and returned 500.
    import ast
    import inspect
    from pathlib import Path

    import orchard.server.routes as routes

    signature = inspect.signature(gather_non_streaming_batch_response)
    calls = 0
    for source in Path(routes.__file__).parent.glob("*.py"):
        for node in ast.walk(ast.parse(source.read_text())):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "gather_non_streaming_batch_response"
            ):
                signature.bind(*node.args, **{kw.arg: kw.value for kw in node.keywords})
                calls += 1
    assert calls >= 2
