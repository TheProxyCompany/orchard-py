from __future__ import annotations

import asyncio
import json
import os
import socket
import threading
import uuid
from pathlib import Path

import pytest

from tests.shared_owner import (
    BuckshotAdmissionGate,
    buckshot_phase,
    observe_request_attempt,
)


def _barrier_path() -> Path:
    return Path("/private/tmp") / f"bs-{os.getpid()}-{uuid.uuid4().hex[:8]}.sock"


def _read_frame(connection: socket.socket) -> dict[str, object]:
    payload = bytearray()
    while b"\n" not in payload:
        chunk = connection.recv(65_536)
        if not chunk:
            raise RuntimeError("protocol connection closed before a full frame")
        payload.extend(chunk)
    return json.loads(payload.partition(b"\n")[0])


@pytest.mark.asyncio
async def test_shared_owner_parks_cases_then_measures_sequential_phase_zero_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _barrier_path()
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(os.fspath(path))
    listener.listen(1)
    frames: list[dict[str, object]] = []
    ready_seen = threading.Event()
    allow_release = threading.Event()

    def coordinate() -> None:
        connection, _ = listener.accept()
        with connection:
            frames.append(_read_frame(connection))
            ready_seen.set()
            allow_release.wait(timeout=2)
            connection.sendall(b"G")
            frames.append(_read_frame(connection))

    thread = threading.Thread(target=coordinate)
    thread.start()
    request_plan = {"case::a": [2, 1], "case::b": [1], "helper::only": []}
    monkeypatch.setenv("ORCHARD_TEST_SHARED_OWNER", "1")
    monkeypatch.setenv("ORCHARD_TEST_BARRIER_SOCKET", os.fspath(path))
    monkeypatch.setenv(
        "ORCHARD_TEST_EXPECTED_PY_REQUEST_PLAN_JSON", json.dumps(request_plan)
    )
    gate = BuckshotAdmissionGate(["model-a", "model-b"], request_plan)
    released = asyncio.Event()
    request_started = asyncio.Event()

    async def run_case_a() -> None:
        await gate.admit("case::a")
        request_started.set()
        await observe_request_attempt()
        await observe_request_attempt()
        released.set()
        with buckshot_phase(1):
            await observe_request_attempt()
        gate.complete("case::a")

    async def run_case_b() -> None:
        await gate.admit("case::b")
        await observe_request_attempt()
        await released.wait()
        gate.complete("case::b")

    async def run_helper() -> None:
        await gate.admit("helper::only")
        gate.complete("helper::only")

    try:
        volley = asyncio.gather(run_case_a(), run_case_b(), run_helper())
        assert await asyncio.to_thread(ready_seen.wait, 2)
        assert not request_started.is_set()
        allow_release.set()
        await volley
        await gate.finish(success=True)
        thread.join(timeout=2)
    finally:
        allow_release.set()
        listener.close()
        path.unlink(missing_ok=True)

    assert not thread.is_alive()
    assert frames[0] == {
        "type": "ready",
        "protocol": 3,
        "role": "orchard-py",
        "models": ["model-a", "model-b"],
        "cases": ["case::a", "case::b", "helper::only"],
        "cases_admitted": 3,
        "request_plan": request_plan,
        "phase_counts": [3, 1],
        "phase0_requests": 3,
        "request_attempts": 0,
        "filtered": 0,
        "skipped": 0,
        "inference_retries": 0,
    }
    assert frames[1]["cases_completed"] == 3
    assert frames[1]["request_attempts"] == 4
    assert frames[1]["phase_counts"] == [3, 1]
    assert frames[1]["inference_retries"] == 0
    assert frames[1]["success"] is True


def test_shared_owner_rejects_parent_plan_disagreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORCHARD_TEST_SHARED_OWNER", "1")
    monkeypatch.setenv(
        "ORCHARD_TEST_EXPECTED_PY_REQUEST_PLAN_JSON", json.dumps({"case::a": [1]})
    )

    with pytest.raises(RuntimeError, match="root owner's manifest"):
        BuckshotAdmissionGate(["model-a"], {"case::a": [2]})


@pytest.mark.asyncio
async def test_shared_owner_requires_barrier_at_case_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_plan = {"case::a": [1]}
    monkeypatch.setenv("ORCHARD_TEST_SHARED_OWNER", "1")
    monkeypatch.setenv(
        "ORCHARD_TEST_EXPECTED_PY_REQUEST_PLAN_JSON", json.dumps(request_plan)
    )
    monkeypatch.delenv("ORCHARD_TEST_BARRIER_SOCKET", raising=False)
    gate = BuckshotAdmissionGate(["model-a"], request_plan)

    with pytest.raises(RuntimeError, match="ORCHARD_TEST_BARRIER_SOCKET"):
        await gate.admit("case::a")
    gate.complete("case::a")
    with pytest.raises(RuntimeError, match="missing request attempts"):
        await gate.finish(success=False)
