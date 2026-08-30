import ctypes
import os

from orchard.engine import multiprocess


def _proc_pidinfo_with_status(status: int):
    def proc_pidinfo(_pid, _flavor, _arg, buffer, size):
        info = ctypes.cast(
            buffer, ctypes.POINTER(multiprocess._ProcBSDShortInfo)
        ).contents
        info.status = status
        return size

    return proc_pidinfo


def test_pid_is_alive_uses_darwin_process_status_without_spawning(monkeypatch):
    monkeypatch.setattr(multiprocess, "platform", "darwin")
    monkeypatch.setattr(
        multiprocess, "_proc_pidinfo", _proc_pidinfo_with_status(status=2)
    )

    assert multiprocess.pid_is_alive(os.getpid())


def test_pid_is_alive_rejects_darwin_zombie(monkeypatch):
    monkeypatch.setattr(multiprocess, "platform", "darwin")
    monkeypatch.setattr(
        multiprocess,
        "_proc_pidinfo",
        _proc_pidinfo_with_status(status=multiprocess._SZOMB),
    )

    assert not multiprocess.pid_is_alive(os.getpid())


def test_failed_darwin_status_probe_defers_to_kill_zero(monkeypatch):
    monkeypatch.setattr(multiprocess, "platform", "darwin")
    monkeypatch.setattr(multiprocess, "_proc_pidinfo", lambda *_args: 0)

    assert multiprocess.pid_is_alive(os.getpid())
