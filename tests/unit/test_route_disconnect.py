import asyncio
import inspect
import json
from functools import partial

import httpx
import pynng
import pytest
from fastapi import FastAPI

from orchard.app.ipc_dispatch import (
    EngineDisconnectedError,
    IPCState,
    QueueRegistration,
)
from orchard.app.model_registry import (
    ModelEntry,
    ModelInfo,
    ModelLoadState,
    ModelRegistry,
)
from orchard.engine.global_context import GlobalContext
from orchard.formatter.formatter import ChatFormatter
from orchard.server.exceptions import InferenceError
from orchard.server.routes.chat import (
    chat_router,
    gather_non_streaming_batch_response,
    stream_response_generator,
)
from orchard.server.routes.embeddings import (
    embeddings_router,
    gather_embedding_response,
)
from orchard.server.routes.responses import (
    gather_non_streaming_response,
    responses_router,
)
from orchard.server.routes.responses import (
    stream_response_generator as stream_responses_generator,
)

GATHERERS = [
    partial(
        gather_non_streaming_batch_response,
        prompt_fanout_counts=[2],
        prompt_final_counts=[2],
    ),
    gather_non_streaming_response,
    gather_embedding_response,
]
STREAMERS = [
    partial(
        stream_response_generator,
        model_name="transport-only",
        expected_sequences=2,
        generation_model="transport-only",
        thinking_flags=[False],
        released_text=True,
        include_usage=True,
    ),
    partial(
        stream_responses_generator,
        response_id="resp_disconnect",
        model_name="transport-only",
    ),
]


async def _disconnected_state(tmp_path, buffered_terminal):
    state = IPCState(GlobalContext())
    queue = asyncio.Queue()
    if buffered_terminal:
        # Exercise the real dispatcher's death detection and terminal callback.
        state.engine_pid_file = tmp_path / "missing-engine.pid"
        state.active_request_queues[42] = QueueRegistration(
            loop=asyncio.get_running_loop(), queue=queue
        )
        with pynng.Sub0() as socket:
            state.response_socket = socket
            await IPCState.run_ipc_listener(state)
            await asyncio.sleep(0)
        state.response_socket = None
        assert queue.qsize() == 1
    else:
        # The death latch can become visible before the terminal callback runs.
        state._disconnect()
    assert state.engine_dead
    return state, queue


@pytest.mark.asyncio
@pytest.mark.parametrize("buffered_terminal", [False, True])
@pytest.mark.parametrize("gather", GATHERERS, ids=["chat", "responses", "embeddings"])
async def test_nonstreaming_disconnect_is_an_inference_error(
    tmp_path, gather, buffered_terminal
):
    state, queue = await _disconnected_state(tmp_path, buffered_terminal)
    with pytest.raises(InferenceError, match="Engine process disconnected") as error:
        await asyncio.wait_for(gather(42, queue, state), 1)
    if not buffered_terminal:
        assert isinstance(error.value.__cause__, EngineDisconnectedError)
    await asyncio.wait_for(queue.join(), 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("buffered_terminal", [False, True])
async def test_multichoice_chat_disconnect_preserves_error_and_finishes(
    tmp_path, buffered_terminal
):
    state, queue = await _disconnected_state(tmp_path, buffered_terminal)
    async with asyncio.timeout(1):
        events = [event async for event in STREAMERS[0](42, queue, state)]
    assert events[-1] == {"data": "[DONE]"}
    chunks = [json.loads(event["data"]) for event in events[:-1]]
    choices = [choice for chunk in chunks for choice in chunk["choices"]]
    assert sum(choice.get("finish_reason") == "error" for choice in choices) == 1
    assert any(
        choice["delta"].get("content") == "Engine process disconnected."
        for choice in choices
    )
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"]["total_tokens"] == 0
    await asyncio.wait_for(queue.join(), 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("buffered_terminal", [False, True])
async def test_responses_disconnect_emits_failed_and_error_before_done(
    tmp_path, buffered_terminal
):
    state, queue = await _disconnected_state(tmp_path, buffered_terminal)
    async with asyncio.timeout(1):
        events = [event async for event in STREAMERS[1](42, queue, state)]
    assert events[-1] == {"data": "[DONE]"}
    payloads = [json.loads(event["data"]) for event in events[:-1]]
    assert [payload["type"] for payload in payloads] == [
        "response.created",
        "response.in_progress",
        "response.failed",
        "error",
    ]
    assert payloads[-2]["response"]["status"] == "failed"
    assert payloads[-1]["error"]["message"] == "Engine process disconnected."
    await asyncio.wait_for(queue.join(), 1)


@pytest.fixture
def queue_on_closed_loop():
    async def bind_queue():
        queue = asyncio.Queue()
        reader = asyncio.create_task(queue.get())
        await asyncio.sleep(0)
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)
        return queue

    return asyncio.run(bind_queue())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "consume",
    GATHERERS + STREAMERS,
    ids=["chat", "responses", "embeddings", "chat-stream", "responses-stream"],
)
async def test_unrelated_runtime_error_is_not_hidden(consume, queue_on_closed_loop):
    state = IPCState(GlobalContext())
    with pytest.raises(RuntimeError, match="bound to a different event loop"):
        result = consume(42, queue_on_closed_loop, state)
        if inspect.isasyncgen(result):
            async for _ in result:
                pass
        else:
            await result


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["send", "inspect", "load"])
@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/chat/completions", {"messages": [{"role": "user", "content": "hello"}]}),
        ("/responses", {"input": "hello"}),
        ("/embeddings", {"input": "hello"}),
    ],
    ids=["chat", "responses", "embeddings"],
)
async def test_http_disconnect_during_submission_or_model_management(
    tmp_path, phase, path, body
):
    context = GlobalContext()
    state = IPCState(context)
    registry = ModelRegistry(state)
    model_id = str(tmp_path)
    if phase != "inspect":
        entry = ModelEntry(
            state=ModelLoadState.READY if phase == "send" else ModelLoadState.LOADING,
            info=ModelInfo(
                model_id=model_id,
                model_path=model_id,
                formatter=ChatFormatter.from_config(model_id, {"model_type": "llama"}),
            ),
        )
        entry.event.set()
        registry._entries[model_id] = entry
        registry._alias_cache[model_id.lower()] = model_id

    app = FastAPI()
    app.include_router(chat_router)
    app.include_router(responses_router)
    app.include_router(embeddings_router)
    app.state.ipc_state = state
    app.state.model_registry = registry
    with pynng.Push0(send_buffer_size=0) as requests, pynng.Req0() as management:
        state.request_socket = requests
        state.management_socket = management
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://orchard.test"
        ) as client:
            pending = asyncio.create_task(
                client.post(path, json={"model": model_id, **body})
            )
            try:
                async with asyncio.timeout(1):
                    while not state._socket_tasks:
                        if pending.done():
                            pytest.fail(f"Request finished before IPC: {await pending}")
                        await asyncio.sleep(0)
                    await asyncio.sleep(0)
                    assert not pending.done()
                    state._disconnect()
                    response = await pending
                assert response.status_code == (502 if phase == "send" else 503)
                assert "Engine process disconnected." in response.json()["detail"]
                assert not state.active_request_queues
            finally:
                state._disconnect()
                if not pending.done():
                    pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)
                assert state.wait_for_inflight_drain(0)
