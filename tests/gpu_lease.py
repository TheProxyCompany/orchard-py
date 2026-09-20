"""The machine-wide GPU lease: one heavy GPU tenant at a time.

Apple's GPU firmware on the M3 Ultra test machine locks up when too many GPU
command queues, from any processes, stream large weight sets at once. It is
device-wide: two processes that are each clean alone hang together. The test
machine is also the GitHub Actions runner, so CI jobs, local test sessions and
benchmarks all meet on one GPU, and nothing else coordinates them.

Every repo that puts test or benchmark work on the GPU implements this same
contract (path, flock, the waiting line, the holder line, the two environment
variables), so keep it in step with the others. Test infrastructure only: the
orchard package never takes the lease.
"""

import fcntl
import os
import sys
import time

LEASE_PATH = "/tmp/proxy-gpu.lease"


def hold_gpu_lease(what: str, path: str = LEASE_PATH) -> int | None:
    """Block until this process is the machine's one heavy GPU tenant.

    `what` names the session in the lease file for whoever waits next. Returns
    the open fd. Nothing closes it, so the lock lasts until the process exits,
    and the kernel drops it if the process dies: no stale lease, no cleanup.
    Returns None without locking when a parent already holds the lease
    (PROXY_GPU_LEASE_HELD; a flock belongs to the open file description, so a
    child taking it again would deadlock) or on the explicit opt-out
    PROXY_GPU_LEASE=0. `path` exists for the unit test.
    """
    if os.environ.get("PROXY_GPU_LEASE") == "0" or os.environ.get(
        "PROXY_GPU_LEASE_HELD"
    ):
        return None
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        # The umask cuts os.open's mode, and another user's session must be
        # able to open the file too.
        os.fchmod(fd, 0o666)
    except PermissionError:
        pass  # another user created it
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        holder = os.pread(fd, 300, 0).decode(errors="replace").strip()
        print(
            f"waiting for the GPU lease ({path}) held by: {holder}",
            file=sys.stderr,
            flush=True,
        )
        fcntl.flock(fd, fcntl.LOCK_EX)  # no timeout: CI waits for a local run
    os.ftruncate(fd, 0)
    since = time.strftime("%Y-%m-%dT%H:%M:%S")
    os.pwrite(fd, f"pid={os.getpid()} {what} since={since}\n".encode(), 0)
    os.environ["PROXY_GPU_LEASE_HELD"] = "1"
    return fd
