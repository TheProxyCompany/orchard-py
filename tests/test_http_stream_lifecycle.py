import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import anyio
import httpx
import pytest
from fastapi import FastAPI

from orchard.app.model_registry import ModelInfo, ModelLoadState
from orchard.formatter.formatter import ChatFormatter
from orchard.server.routes._common import managed_stream_session
from orchard.server.routes.chat import chat_router
from orchard.server.routes.completions import completions_router
from orchard.server.routes.responses import responses_router


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "candidates", "finals", "native_error"),
    [
        ("/chat/completions", 1, 0, False),
        ("/chat/completions", 2, 1, False),
        ("/chat/completions", 2, 2, False),
        ("/responses", 1, 0, False),
        ("/responses", 1, 1, False),
        ("/responses", 1, 1, True),
    ],
)
async def test_sse_cancels_only_requests_missing_a_native_terminal(
    path, candidates, finals, native_error
):
    model = "meta-llama/Llama-3.1-8B-Instruct"
    info = ModelInfo(
        model,
        "/fixture/llama",
        ChatFormatter.from_config(
            "/fixture/llama",
            {"model_type": "llama", "architectures": ["LlamaForCausalLM"]},
        ),
    )
    state = SimpleNamespace(
        active_request_queues={},
        response_channel_id=1,
        lossless_responses=True,
        get_next_request_id=AsyncMock(return_value=23),
        cancel_request=AsyncMock(),
    )

    async def submit(_payload):
        queue = state.active_request_queues[23].queue
        for candidate in range(finals):
            delta = {
                "sequence_id": candidate + 1,
                "prompt_index": 0,
                "candidate_index": candidate,
                "is_final_delta": True,
                "finish_reason": "error" if native_error else "stop",
                "tokens": [],
            }
            if native_error:
                delta["error"] = "fixture inference error"
            queue.put_nowait(delta)

    async def next_delta(queue):
        if queue.empty():
            raise TimeoutError
        return await queue.get()

    state.send_request = submit
    state.next_delta = next_delta
    app = FastAPI()
    app.include_router(chat_router)
    app.include_router(responses_router)
    app.state.ipc_state = state
    app.state.model_registry = SimpleNamespace(
        schedule_http_model=AsyncMock(return_value=(ModelLoadState.READY, model)),
        get_if_ready=lambda _model: info,
    )
    app.state.model_directories = ()
    payload = {"model": model, "stream": True, "temperature": 0}
    if path == "/chat/completions":
        payload.update(n=candidates, messages=[{"role": "user", "content": "Hello"}])
    else:
        payload["input"] = "Hello"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(path, json=payload)

    assert response.status_code == 200
    assert "data: [DONE]" in response.text
    if path == "/responses" and (native_error or not finals):
        assert "event: response.failed" in response.text
    if finals < candidates:
        state.cancel_request.assert_awaited_once_with(23)
    else:
        state.cancel_request.assert_not_awaited()
    assert not state.active_request_queues


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "inputs"),
    [
        ("/chat/completions", {"messages": [{"role": "user", "content": "Hello"}]}),
        ("/responses", {"input": "Hello"}),
        ("/completions", {"prompt": "Hello"}),
    ],
)
async def test_non_streaming_disconnect_releases_the_submitted_request(path, inputs):
    model = "meta-llama/Llama-3.1-8B-Instruct"
    info = ModelInfo(
        model, "/fixture/llama",
        ChatFormatter.from_config("/fixture/llama", {"model_type": "llama", "architectures": ["LlamaForCausalLM"]}),
    )
    submitted = asyncio.Event()
    state = SimpleNamespace(
        active_request_queues={}, response_channel_id=7, lossless_responses=True,
        get_next_request_id=AsyncMock(return_value=23), cancel_request=AsyncMock(),
        send_request=AsyncMock(side_effect=lambda _: submitted.set()), next_delta=lambda queue: queue.get(),
    )
    app = FastAPI()
    for router in (chat_router, responses_router, completions_router):
        app.include_router(router)
    app.state.ipc_state = state
    app.state.model_directories = ()
    app.state.model_registry = SimpleNamespace(
        schedule_http_model=AsyncMock(return_value=(ModelLoadState.READY, model)),
        get_if_ready=lambda _: info,
    )
    body = json.dumps({"model": model, "stream": False, **inputs}).encode()
    messages = iter([{"type": "http.request", "body": body, "more_body": False}])

    async def receive():
        message = next(messages, None)
        if message is not None:
            return message
        await submitted.wait()
        return {"type": "http.disconnect"}

    sent = AsyncMock()
    scope = {"type": "http", "method": "POST", "path": path, "query_string": b"",
             "headers": [(b"content-type", b"application/json")], "http_version": "1.1", "scheme": "http"}
    await asyncio.wait_for(app(scope, receive, sent), timeout=2)
    assert sent.await_args_list[0].args[0]["status"] == 499
    state.cancel_request.assert_awaited_once_with(23)
    assert not state.active_request_queues
