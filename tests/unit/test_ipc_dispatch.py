import asyncio
import json
import os
import struct
import tempfile
import threading
from types import SimpleNamespace

import pynng
import pynng._aio
import pytest

from orchard import defaults
from orchard.app.ipc_dispatch import (
    EVENT_TOPIC_PREFIX,
    EngineDisconnectedError,
    IPCState,
)
from orchard.engine.global_context import GlobalContext
from orchard.engine.inference_engine import InferenceEngine
from orchard.ipc.serialization import _build_request_payload


@pytest.mark.asyncio
async def test_cancel_request_keeps_its_channel_when_request_ids_collide():
    with tempfile.TemporaryDirectory(prefix="pycancel-", dir="/tmp") as directory:
        request_url = f"ipc://{directory}/requests"
        management_url = f"ipc://{directory}/management"
        contexts = [GlobalContext(), GlobalContext()]
        states = [IPCState(context) for context in contexts]
        cancel = None
        with (
            pynng.Pull0(listen=request_url, recv_timeout=1000) as requests,
            pynng.Rep0(listen=management_url, recv_timeout=1000) as management,
        ):
            try:
                routes = []
                for state in states:
                    state.response_channel_id = (
                        InferenceEngine.generate_response_channel_id()
                    )
                    state.request_socket = pynng.Push0(
                        dial=request_url, send_timeout=1000
                    )
                    state.management_socket = pynng.Req0(
                        dial=management_url, send_timeout=1000
                    )
                    request_id = await state.get_next_request_id()
                    await state.send_request(
                        _build_request_payload(
                            request_id=request_id,
                            model_id="transport-only",
                            model_path=directory,
                            request_type="generation",
                            response_channel_id=state.response_channel_id,
                            prompts=[{"prompt": "no model is loaded"}],
                        )
                    )
                    frame = await asyncio.wait_for(requests.arecv(), 1)
                    length = struct.unpack_from("<I", frame)[0]
                    routes.append(json.loads(frame[4 : 4 + length]))

                assert routes[0]["request_id"] == routes[1]["request_id"] == 1
                channels = [route["response_channel_id"] for route in routes]
                assert all(channels) and channels[0] != channels[1]
                for state, route, other in zip(
                    states, routes, reversed(routes), strict=True
                ):
                    cancel = asyncio.create_task(
                        state.cancel_request(route["request_id"])
                    )
                    command = json.loads(await asyncio.wait_for(management.arecv(), 1))
                    await management.asend(json.dumps({"status": "accepted"}).encode())
                    assert await asyncio.wait_for(cancel, 1) == {"status": "accepted"}
                    print(
                        json.dumps(
                            {
                                "request_channel": route["response_channel_id"],
                                "other_channel": other["response_channel_id"],
                                "cancel": command,
                            }
                        )
                    )
                    assert command == {
                        "type": "cancel_request",
                        "request_id": 1,
                        "response_channel_id": route["response_channel_id"],
                    }
                    assert (
                        command["response_channel_id"] != other["response_channel_id"]
                    )
            finally:
                if cancel is not None and not cancel.done():
                    cancel.cancel()
                    await asyncio.gather(cancel, return_exceptions=True)
                for state in states:
                    assert state.wait_for_inflight_drain(1)
                    if state.request_socket is not None:
                        state.request_socket.close()
                    if state.management_socket is not None:
                        state.management_socket.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_mode", ["disconnect", "cancel"])
async def test_management_send_completion_preserves_cancellation(
    monkeypatch, exit_mode
):
    state = IPCState(GlobalContext())
    original = pynng._aio.AIOHelper._aio_helper_map["asyncio"]
    injected = False
    request = None

    def completion_helper(aio):
        nonlocal injected
        awaitable, completed = original(aio)
        if injected:
            return awaitable, completed
        injected = True
        loop = asyncio.get_running_loop()

        def completed_send():
            # The real NNG send has succeeded, but its Python waiter has not
            # resumed. Deliver cancellation at this completion boundary.
            callback = (
                state._disconnect if exit_mode == "disconnect" else request.cancel
            )
            loop.call_soon_threadsafe(callback)
            completed()

        return awaitable, completed_send

    with tempfile.TemporaryDirectory(prefix="pycancel-", dir="/tmp") as directory:
        url = f"ipc://{directory}/management"
        with pynng.Rep0(listen=url), pynng.Req0(dial=url) as socket:
            state.management_socket = socket
            monkeypatch.setitem(
                pynng._aio.AIOHelper._aio_helper_map, "asyncio", completion_helper
            )
            request = asyncio.create_task(
                state.send_management_command({"type": "cancel_request"}, timeout=None)
            )
            try:
                _, pending = await asyncio.wait([request], timeout=1)
                assert injected
                assert not pending, "Cancelled send started a management receive"
                expected = (
                    EngineDisconnectedError
                    if exit_mode == "disconnect"
                    else asyncio.CancelledError
                )
                with pytest.raises(expected):
                    request.result()
                assert state.wait_for_inflight_drain(0)
            finally:
                if not request.done():
                    request.cancel()
                await asyncio.gather(request, return_exceptions=True)


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


@pytest.mark.asyncio
async def test_delta_reader_distinguishes_disconnect_from_live_silence(monkeypatch):
    state = IPCState(GlobalContext())
    queue = asyncio.Queue()
    state._disconnect()
    with pytest.raises(EngineDisconnectedError, match="Engine process disconnected"):
        await state.next_delta(queue)

    # A terminal callback may arrive after the death latch. Its real request
    # identity and payload must still be delivered unchanged.
    terminal = {"request_id": 42, "is_final_delta": True, "finish_reason": "error"}
    queue.put_nowait(terminal)
    assert await state.next_delta(queue) is terminal

    live_state = IPCState(GlobalContext())
    monkeypatch.setattr(defaults, "DELTA_TIMEOUT_S", 0.01)
    with pytest.raises(TimeoutError, match="Timed out waiting"):
        await live_state.next_delta(asyncio.Queue())


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
