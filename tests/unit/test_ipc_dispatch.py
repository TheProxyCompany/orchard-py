import os

from orchard.app.ipc_dispatch import IPCState
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
