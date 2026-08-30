import asyncio
import json
import os
import threading
import uuid
from types import SimpleNamespace

import pynng
import pytest

from orchard.app.ipc_dispatch import (
    EVENT_TOPIC_PREFIX,
    IPCState,
    QueueRegistration,
)
from orchard.engine.global_context import GlobalContext
from orchard.engine.inference_engine import InferenceEngine


def test_engine_process_is_alive_reads_pid_file(tmp_path):
    ipc_state = IPCState(GlobalContext())
    pid_file = tmp_path / "engine.pid"
    pid_file.write_text(f"{os.getpid()}\n", encoding="utf-8")

    ipc_state.engine_pid_file = pid_file

    assert ipc_state.engine_process_is_alive()


def test_engine_process_is_alive_handles_missing_pid_file(tmp_path):
    ipc_state = IPCState(GlobalContext())
    ipc_state.engine_pid_file = tmp_path / "missing.pid"

    assert not ipc_state.engine_process_is_alive()


def test_handle_engine_event_routes_model_load_failed():
    model_registry = SimpleNamespace()
    captured: dict | None = None

    def handle_model_load_failed(payload: dict) -> None:
        nonlocal captured
        captured = payload

    model_registry.handle_model_load_failed = handle_model_load_failed

    ctx = GlobalContext()
    ctx.model_registry = model_registry
    ipc_state = IPCState(ctx)

    payload = {
        "event": "model_load_failed",
        "model_id": "broken/model",
        "error": "missing shard",
    }
    message = (
        EVENT_TOPIC_PREFIX
        + b"model_load_failed"
        + b"\x00"
        + json.dumps(payload).encode("utf-8")
    )

    ipc_state.handle_engine_event(message)

    assert captured == payload


def test_socket_op_refuses_new_ops_once_shutdown_requested():
    ipc_state = IPCState(GlobalContext())
    ipc_state.shutdown_requested = True

    with pytest.raises(RuntimeError, match="shutdown in progress"):
        with ipc_state.socket_op():
            pass


def test_wait_for_inflight_drain_tracks_socket_ops():
    ipc_state = IPCState(GlobalContext())

    assert ipc_state.wait_for_inflight_drain(0)
    with ipc_state.socket_op():
        assert not ipc_state.wait_for_inflight_drain(0.05)
    assert ipc_state.wait_for_inflight_drain(0)


@pytest.mark.asyncio
async def test_next_delta_waits_for_owned_request_until_delivery():
    ipc_state = IPCState(GlobalContext())
    queue: asyncio.Queue = asyncio.Queue()

    waiter = asyncio.create_task(ipc_state.next_delta(queue))
    await asyncio.sleep(0.01)
    assert not waiter.done()

    payload = {"request_id": 7, "content": "ready"}
    await queue.put(payload)

    assert await waiter == payload


@pytest.mark.asyncio
async def test_listener_multiplexes_flow_controlled_responses_and_events():
    suffix = uuid.uuid4().hex
    response_url = f"inproc://response-{suffix}"
    event_url = f"inproc://events-{suffix}"

    ctx = GlobalContext()
    ipc_state = IPCState(ctx)
    ipc_state.response_channel_id = 0xCAFE
    ipc_state.response_socket = pynng.Pull0(listen=response_url)
    event_sender = pynng.Pub0(listen=event_url)
    ipc_state.event_socket = pynng.Sub0()
    ipc_state.event_socket.subscribe(b"resp:cafe:")
    ipc_state.event_socket.subscribe(EVENT_TOPIC_PREFIX)
    ipc_state.event_socket.dial(event_url, block=True)

    response_sender = pynng.Push0(dial=response_url, block_on_dial=True)

    request_queue: asyncio.Queue = asyncio.Queue()
    ipc_state.active_request_queues[7] = QueueRegistration(
        loop=asyncio.get_running_loop(),
        queue=request_queue,
    )
    legacy_request_queue: asyncio.Queue = asyncio.Queue()
    ipc_state.active_request_queues[8] = QueueRegistration(
        loop=asyncio.get_running_loop(),
        queue=legacy_request_queue,
    )
    listener = asyncio.create_task(IPCState.run_ipc_listener(ipc_state))

    try:
        await asyncio.sleep(0.05)
        response_prefix = b"resp:cafe:"
        response_payload = json.dumps(
            {"request_id": 7, "content": "delta", "is_final_delta": False}
        ).encode("utf-8")
        await response_sender.asend(response_prefix + response_payload)

        telemetry_payload = json.dumps({"gpu_busy": True}).encode("utf-8")
        await event_sender.asend(
            EVENT_TOPIC_PREFIX + b"telemetry\x00" + telemetry_payload
        )
        legacy_payload = json.dumps(
            {"request_id": 8, "content": "legacy", "is_final_delta": True}
        ).encode("utf-8")
        await event_sender.asend(response_prefix + legacy_payload)

        assert await asyncio.wait_for(request_queue.get(), timeout=1.0) == {
            "request_id": 7,
            "content": "delta",
            "is_final_delta": False,
        }
        assert await asyncio.wait_for(legacy_request_queue.get(), timeout=1.0) == {
            "request_id": 8,
            "content": "legacy",
            "is_final_delta": True,
        }
        deadline = asyncio.get_running_loop().time() + 1.0
        while ctx.last_telemetry != {"gpu_busy": True}:
            if asyncio.get_running_loop().time() >= deadline:
                pytest.fail("event socket did not deliver telemetry")
            await asyncio.sleep(0.01)
        ipc_state.active_request_queues.pop(7)
        ipc_state.active_request_queues.pop(8)
    finally:
        ipc_state.shutdown_requested = True
        await asyncio.wait_for(listener, timeout=3.0)
        response_sender.close()
        event_sender.close()
        ipc_state.response_socket.close()
        ipc_state.event_socket.close()


def test_socket_close_waits_for_inflight_ops_to_drain():
    ctx = GlobalContext()
    ipc_state = IPCState(ctx)
    ctx.ipc_state = ipc_state

    order: list[str] = []
    ipc_state.request_socket = SimpleNamespace(close=lambda: order.append("closed"))

    op = ipc_state.socket_op()
    op.__enter__()
    timer = threading.Timer(
        0.2, lambda: (order.append("op_done"), op.__exit__(None, None, None))
    )
    timer.start()

    ipc_state.shutdown_requested = True
    InferenceEngine._close_sockets_if_dispatcher_stopped(ctx)
    timer.join()

    assert order == ["op_done", "closed"]
    assert ipc_state.request_socket is None
