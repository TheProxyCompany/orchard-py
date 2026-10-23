"""The GPU lease (tests/gpu_lease.py) serializes sessions on one machine.

Stub sessions only, no engine and no GPU. The helper's sessions are
subprocesses that take the lease on a temp path and hold it until their stdin
closes. The fixture's sessions are pytest subprocesses that load
tests/conftest.py with the lease and the engine class replaced by recorders.
"""

import json
import os
import re
import select
import subprocess
import sys
import time
from pathlib import Path

HELPER_DIR = str(Path(__file__).parent.parent)

SESSION = """
import sys
sys.path.insert(0, sys.argv[1])
from gpu_lease import hold_gpu_lease
fd = hold_gpu_lease(sys.argv[3], sys.argv[2])
print("skipped" if fd is None else "holding", flush=True)
sys.stdin.read()
"""

HOLDER_WITH_CHILD = """
import subprocess, sys
sys.path.insert(0, sys.argv[1])
from gpu_lease import hold_gpu_lease
assert hold_gpu_lease("stub-parent", sys.argv[2]) is not None
child = subprocess.run(
    [sys.executable, "-c", sys.argv[3], sys.argv[1], sys.argv[2], "stub-child"],
    stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30,
)
print(child.stdout.strip(), child.stderr.strip(), flush=True)
"""


def _env(**extra: str) -> dict[str, str]:
    # The unit tests may themselves run under a lease holder or an opt-out.
    env = {k: v for k, v in os.environ.items() if not k.startswith("PROXY_GPU_LEASE")}
    return env | extra


def _session(lease: Path, what: str, *, hold: bool, **env: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", SESSION, HELPER_DIR, str(lease), what],
        stdin=subprocess.PIPE if hold else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_env(**env),
    )


def _holder(lease: Path) -> subprocess.Popen:
    holder = _session(lease, "stub-a", hold=True)
    assert holder.stdout.readline().strip() == "holding", holder.stderr.read()
    return holder


def test_two_sessions_run_one_after_the_other(tmp_path: Path) -> None:
    lease = tmp_path / "gpu.lease"
    with _holder(lease) as first, _session(lease, "stub-b", hold=False) as second:
        try:
            holder_line = lease.read_text().strip()
            assert re.fullmatch(
                rf"pid={first.pid} stub-a since=\d{{4}}-\d\d-\d\dT\d\d:\d\d:\d\d",
                holder_line,
            )

            ready, _, _ = select.select([second.stderr], [], [], 30)
            assert ready, "the second session printed no waiting line"
            assert second.stderr.readline().strip() == (
                f"waiting for the GPU lease ({lease}) held by: {holder_line}"
            )
            time.sleep(0.3)
            assert second.poll() is None, "the second session did not wait"

            first.stdin.close()
            first.wait(timeout=30)
            out, err = second.communicate(timeout=30)
            assert (out.strip(), err, second.returncode) == ("holding", "", 0)
            assert lease.read_text().startswith(f"pid={second.pid} stub-b since=")
        finally:
            first.kill()
            second.kill()


def test_child_of_a_holder_does_not_block(tmp_path: Path) -> None:
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            HOLDER_WITH_CHILD,
            HELPER_DIR,
            str(tmp_path / "gpu.lease"),
            SESSION,
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env=_env(),
    )
    assert (done.stdout.strip(), done.stderr, done.returncode) == ("skipped", "", 0)


def test_opt_out_does_not_block(tmp_path: Path) -> None:
    lease = tmp_path / "gpu.lease"
    with _holder(lease) as first:
        try:
            with _session(lease, "stub-b", hold=False, PROXY_GPU_LEASE="0") as second:
                out, err = second.communicate(timeout=30)
            assert (out.strip(), err, second.returncode) == ("skipped", "", 0)
            assert first.poll() is None, "the holder exited early: nothing was proved"
        finally:
            first.kill()


# Loaded with -p ahead of tests/conftest.py. Each event is written with whether
# another session's engine log is still in the log directory at that moment.
RECORDERS = """
import json
import os
import sys
from pathlib import Path

import orchard.engine.inference_engine as engine_module

EVENTS = Path(__file__).with_name("events.jsonl")
OTHER_LOG = Path(os.environ["ORCHARD_TEST_LOG_DIR"]) / "engine.test.log"


def record(event):
    with EVENTS.open("a") as f:
        f.write(json.dumps([event, OTHER_LOG.exists()]) + "\\n")


class RecordedEngine:
    def __init__(self, **kwargs):
        record("engine constructed")

    def close(self):
        pass

    @staticmethod
    def shutdown(timeout=15.0):
        record("running engine stopped")
        return True


def recorded_lease(what):
    print("the waiting line goes here", file=sys.stderr, flush=True)
    record("lease requested: " + what)
    return 0


engine_module.InferenceEngine = RecordedEngine  # before conftest imports it
import conftest

conftest.hold_gpu_lease = recorded_lease
"""

FIXTURE_TESTS = """
def test_with_the_engine(engine):
    pass


def test_without_an_engine():
    pass
"""


def _pytest_session(
    tmp_path: Path, test: str
) -> tuple[list[list], Path, subprocess.CompletedProcess]:
    (tmp_path / "lease_recorders.py").write_text(RECORDERS)
    (tmp_path / "test_fixture.py").write_text(FIXTURE_TESTS)
    other_log = tmp_path / "logs" / "engine.test.log"
    other_log.parent.mkdir()
    other_log.write_text("the engine log of a session that holds the lease\n")
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *("-p", "lease_recorders", "-p", "conftest", "-p", "no:cacheprovider"),
            "-q",
            f"test_fixture.py::{test}",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env=_env(
            PYTHONPATH=os.pathsep.join([str(tmp_path), HELPER_DIR]),
            # Whatever the recorders miss must find no engine and no real logs.
            ORCHARD_CACHE_ROOT=str(tmp_path / "cache"),
            ORCHARD_TEST_LOG_DIR=str(other_log.parent),
        ),
    )
    assert done.returncode == 0, done.stdout + done.stderr
    events = tmp_path / "events.jsonl"
    lines = events.read_text().splitlines() if events.exists() else []
    return [json.loads(line) for line in lines], other_log, done


def test_engine_fixture_takes_the_lease_before_it_touches_anything(
    tmp_path: Path,
) -> None:
    events, _, done = _pytest_session(tmp_path, "test_with_the_engine")
    lease, *rest = events
    assert lease[0].startswith("lease requested: orchard-py pytest -p "), events
    # The other session's engine and its log outlive the wait for the lease,
    # and the clean slate is there before this session's engine is.
    assert [lease[1], *rest] == [
        True,
        ["running engine stopped", True],
        ["engine constructed", False],
    ]
    # Both reach the terminal or the CI log under the default output capture:
    # whatever the lease prints while it waits, and the line the gate reads.
    assert "the waiting line goes here" in done.stderr
    acquired_line = r"\[gpu-lease\] acquired \d{4}-\d\d-\d\d \d\d:\d\d:\d\d$"
    assert re.search(acquired_line, done.stdout, re.MULTILINE), done.stdout


def test_session_without_the_engine_takes_no_lease_and_stops_nothing(
    tmp_path: Path,
) -> None:
    events, other_log, done = _pytest_session(tmp_path, "test_without_an_engine")
    assert events == []
    assert other_log.exists()
    assert "[gpu-lease]" not in done.stdout
