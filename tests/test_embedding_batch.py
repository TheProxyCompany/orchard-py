import asyncio
import struct
from types import SimpleNamespace

import pytest

from orchard.server.exceptions import InferenceError
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
