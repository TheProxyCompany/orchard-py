from __future__ import annotations

from pathlib import Path

import pytest
from filelock import FileLock

import orchard.engine.inference_engine as inference_engine_module
from orchard.engine.fetch import FetchError, get_engine_path
from orchard.engine.inference_engine import InferenceEngine
from orchard.engine.io import EnginePaths


def _make_engine_paths(tmp_path: Path) -> EnginePaths:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return EnginePaths(
        cache_dir=cache_dir,
        ready_file=cache_dir / "engine.ready",
        pid_file=cache_dir / "engine.pid",
        lock_file=cache_dir / "engine.lock",
        client_log_file=cache_dir / "client.log",
        engine_log_file=cache_dir / "engine.log",
    )


def _make_engine(tmp_path: Path, engine_bin: Path) -> InferenceEngine:
    engine = InferenceEngine.__new__(InferenceEngine)
    paths = _make_engine_paths(tmp_path)
    engine._paths = paths
    engine._lock = FileLock(str(paths.lock_file), timeout=1.0)
    engine._startup_timeout = 1.0
    engine._engine_bin = engine_bin
    engine._lease_active = False
    engine._closed = False
    engine._launch_process = None
    engine.engine_log_path = paths.engine_log_file
    return engine


def test_get_engine_path_requires_present_explicit_local_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    local_build = tmp_path / "missing-build"
    expected_path = local_build / "bin" / "proxy_inference_engine"
    monkeypatch.setenv("PIE_LOCAL_BUILD", str(local_build))

    with pytest.raises(FetchError, match=str(expected_path)):
        get_engine_path()


def test_acquire_lease_restarts_foreign_engine_when_local_build_is_explicit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected_engine = tmp_path / "expected" / "bin" / "proxy_inference_engine"
    expected_engine.parent.mkdir(parents=True)
    expected_engine.touch()
    foreign_engine = tmp_path / "foreign" / "bin" / "proxy_inference_engine"
    foreign_engine.parent.mkdir(parents=True)
    foreign_engine.touch()

    engine = _make_engine(tmp_path, expected_engine)
    events: list[tuple[str, int | None]] = []

    monkeypatch.setenv("PIE_LOCAL_BUILD", str(expected_engine.parent.parent))
    monkeypatch.setattr(inference_engine_module, "read_pid_file", lambda _: 123)
    monkeypatch.setattr(inference_engine_module, "pid_is_alive", lambda _: True)
    monkeypatch.setattr(
        inference_engine_module, "_process_executable_path", lambda _: foreign_engine
    )
    monkeypatch.setattr(
        engine, "_stop_engine_locked", lambda pid: events.append(("stop", pid))
    )
    monkeypatch.setattr(
        engine, "_launch_engine_locked", lambda: events.append(("launch", None))
    )
    monkeypatch.setattr(engine, "_wait_for_engine_ready", lambda: 456)
    monkeypatch.setattr(
        InferenceEngine, "initialize_global_context", staticmethod(lambda ctx, paths: True)
    )
    monkeypatch.setattr(
        InferenceEngine,
        "shutdown_global_context",
        staticmethod(lambda ctx, decrement_ref=True: None),
    )
    monkeypatch.setattr(
        engine,
        "_send_client_lifecycle_command",
        lambda command: events.append((command, None)) or {},
    )

    engine._acquire_lease_and_init_global_context()

    assert events == [("stop", 123), ("launch", None), ("client_register", None)]
    assert engine._lease_active is True


def test_acquire_lease_reuses_matching_engine_when_local_build_is_explicit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected_engine = tmp_path / "expected" / "bin" / "proxy_inference_engine"
    expected_engine.parent.mkdir(parents=True)
    expected_engine.touch()

    engine = _make_engine(tmp_path, expected_engine)
    events: list[tuple[str, int | None]] = []

    monkeypatch.setenv("PIE_LOCAL_BUILD", str(expected_engine.parent.parent))
    monkeypatch.setattr(inference_engine_module, "read_pid_file", lambda _: 123)
    monkeypatch.setattr(inference_engine_module, "pid_is_alive", lambda _: True)
    monkeypatch.setattr(
        inference_engine_module, "_process_executable_path", lambda _: expected_engine
    )
    monkeypatch.setattr(
        engine, "_stop_engine_locked", lambda pid: events.append(("stop", pid))
    )
    monkeypatch.setattr(
        engine, "_launch_engine_locked", lambda: events.append(("launch", None))
    )
    monkeypatch.setattr(
        InferenceEngine, "initialize_global_context", staticmethod(lambda ctx, paths: True)
    )
    monkeypatch.setattr(
        engine,
        "_send_client_lifecycle_command",
        lambda command: events.append((command, None)) or {},
    )

    engine._acquire_lease_and_init_global_context()

    assert events == [("client_register", None)]
    assert engine._lease_active is True
