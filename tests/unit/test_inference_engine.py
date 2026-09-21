from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from filelock import FileLock

import orchard.engine.inference_engine as inference_engine_module
from orchard.app.ipc_dispatch import IPCState
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
        InferenceEngine,
        "initialize_global_context",
        staticmethod(lambda ctx, paths: True),
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
        InferenceEngine,
        "initialize_global_context",
        staticmethod(lambda ctx, paths: True),
    )
    monkeypatch.setattr(
        engine,
        "_send_client_lifecycle_command",
        lambda command: events.append((command, None)) or {},
    )

    engine._acquire_lease_and_init_global_context()

    assert events == [("client_register", None)]
    assert engine._lease_active is True


def _acquire_lease(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    engine_running: bool,
    failed_registrations: int = 0,
) -> IPCState:
    """Runs the lease logic with the engine process stubbed out and returns the
    IPC state it set up."""
    engine = _make_engine(tmp_path, tmp_path / "proxy_inference_engine")
    ctx = inference_engine_module.global_context
    monkeypatch.setattr(ctx, "ipc_state", None)
    monkeypatch.delenv("PIE_LOCAL_BUILD", raising=False)
    monkeypatch.setattr(
        inference_engine_module,
        "read_pid_file",
        lambda _: 123 if engine_running else None,
    )
    monkeypatch.setattr(inference_engine_module, "pid_is_alive", lambda _: True)
    monkeypatch.setattr(inference_engine_module, "_pid_is_engine", lambda _: True)
    monkeypatch.setattr(engine, "_stop_engine_locked", lambda pid: None)
    monkeypatch.setattr(engine, "_launch_engine_locked", lambda: None)
    monkeypatch.setattr(engine, "_wait_for_engine_ready", lambda: 456)

    def initialize_global_context(ctx, paths) -> bool:
        ctx.ipc_state = IPCState(ctx)
        return True

    monkeypatch.setattr(
        InferenceEngine,
        "initialize_global_context",
        staticmethod(initialize_global_context),
    )
    monkeypatch.setattr(
        InferenceEngine,
        "shutdown_global_context",
        staticmethod(lambda ctx, decrement_ref=True: None),
    )
    registrations = 0

    def register(command: str) -> dict:
        nonlocal registrations
        registrations += 1
        if registrations <= failed_registrations:
            raise RuntimeError("the engine is shutting down")
        return {}

    monkeypatch.setattr(engine, "_send_client_lifecycle_command", register)

    engine._acquire_lease_and_init_global_context()

    assert ctx.ipc_state is not None
    return ctx.ipc_state


def test_an_engine_this_process_launched_may_be_asked_for_lossless_responses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ipc_state = _acquire_lease(monkeypatch, tmp_path, engine_running=False)

    assert ipc_state.launched_engine
    assert ipc_state.lossless_responses


def test_a_running_engine_is_not_asked_until_it_advertises_lossless_responses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ipc_state = _acquire_lease(monkeypatch, tmp_path, engine_running=True)

    assert not ipc_state.launched_engine
    assert not ipc_state.lossless_responses


def test_an_engine_launched_to_replace_a_dying_one_counts_as_launched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ipc_state = _acquire_lease(
        monkeypatch, tmp_path, engine_running=True, failed_registrations=1
    )

    assert ipc_state.launched_engine


@pytest.mark.asyncio
async def test_load_models_loads_unique_models_concurrently() -> None:
    engine = InferenceEngine.__new__(InferenceEngine)
    events: list[tuple[str, str]] = []
    active = 0
    max_active = 0

    async def load_model(model_id: str) -> None:
        nonlocal active, max_active
        events.append(("start", model_id))
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0)
        active -= 1
        events.append(("finish", model_id))

    engine.load_model = load_model

    await engine.load_models(["first", "second", "first"])

    assert events.count(("start", "first")) == 1
    assert events.count(("start", "second")) == 1
    assert max_active == 2
