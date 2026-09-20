"""The GPU lease (tests/gpu_lease.py) serializes sessions on one machine.

Stub sessions only: each is a subprocess that takes the lease on a temp path
and then holds it until its stdin closes. No engine, no GPU.
"""

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
