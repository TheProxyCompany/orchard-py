import json
import os
import threading
from types import SimpleNamespace

import pytest

from orchard.app.ipc_dispatch import EVENT_TOPIC_PREFIX, IPCState
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
