import json
import os
from types import SimpleNamespace

from orchard.app.ipc_dispatch import EVENT_TOPIC_PREFIX, IPCState
from orchard.engine.global_context import GlobalContext


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
