#!/usr/bin/env bash
# Run a command as this machine's one heavy GPU tenant:
#   scripts/gpu_lease.sh <command> [args...]
#
# The unit suite loads real model weights and runs them on the GPU. Test runs that
# share a machine (engineers, the git hooks, the CI runner) reset the GPU under each
# other and lose tests to "InnocentVictim" errors, so each run takes the machine's
# GPU lease first. The contract is the same in every Proxy repo:
#   - an exclusive flock on /tmp/proxy-gpu.lease, held until the command exits. The
#     kernel drops it when the process dies: no stale lease, no cleanup.
#   - if it is taken, one "waiting" line on stderr, then block. No timeout.
#   - the holder writes "pid=<pid> <what> since=<time>" into the file and sets
#     PROXY_GPU_LEASE_HELD=1, so its children do not take the lease again (they
#     would deadlock: the lock belongs to the open file, not to the process).
#   - PROXY_GPU_LEASE=0 opts out, to run two tenants on purpose.
#
# Wrap only the step that uses the GPU; compiling must not hold the lease.
# build.sh --run-tests and dev_cycle.sh run ctest through this wrapper, and
# scripts/repro_multi_model_residency.py puts itself under it. Nothing launches the
# GPU binaries that are run by hand (the perf and invariance gates in
# tests/invariance, pie_cache_persistence_bench), so run them through it yourself:
#   scripts/gpu_lease.sh ./build/tests/invariance/pie_gemm_perf [args...]
# "--lease-file <path>" as the first argument is for tests/test_gpu_lease.sh only.
set -euo pipefail

LEASE_FILE="/tmp/proxy-gpu.lease"
if [[ "${1:-}" == "--lease-file" ]]; then
    LEASE_FILE="$2"
    shift 2
fi
[[ $# -gt 0 ]] || {
    echo "Usage: scripts/gpu_lease.sh <command> [args...]" >&2
    echo "  for a GPU binary run by hand: scripts/gpu_lease.sh ./build/tests/invariance/pie_gemm_perf" >&2
    exit 2
}

# python3 because macOS has no flock(1). It execs the command, so the locked file
# stays open in the command and the lease lasts exactly as long as the command.
exec python3 -c '
import fcntl, os, sys, time

path, cmd = sys.argv[1], sys.argv[2:]
if os.environ.get("PROXY_GPU_LEASE") != "0" and not os.environ.get("PROXY_GPU_LEASE_HELD"):
    umask = os.umask(0)  # mode 0666 whatever the umask: other users of the machine lock it too
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    os.umask(umask)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        holder = os.pread(fd, 300, 0).decode(errors="replace").strip()
        print(f"waiting for the GPU lease ({path}) held by: {holder}", file=sys.stderr, flush=True)
        fcntl.flock(fd, fcntl.LOCK_EX)
    # Waiters read the first 300 bytes of the file: keep the whole line inside them.
    what = ("proxy-inference-engine " + " ".join(cmd))[:200]
    since = time.strftime("%Y-%m-%dT%H:%M:%S")
    os.ftruncate(fd, 0)
    os.pwrite(fd, f"pid={os.getpid()} {what} since={since}\n".encode(), 0)
    os.set_inheritable(fd, True)  # survive the exec below
    os.environ["PROXY_GPU_LEASE_HELD"] = "1"
os.execvp(cmd[0], cmd)
' "${LEASE_FILE}" "$@"
