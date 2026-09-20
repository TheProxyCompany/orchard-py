#!/usr/bin/env bash
# Buckshot N-consecutive-runs gate.
#
# Runs the full buckshot matrix N times, each in a fresh pytest session (fresh
# engine process, full hydration), aborting on the first red. A single failure
# resets the streak by definition — rerun the whole gate. Per-run wall clock,
# per-suite timings, the engine log, who else was on the GPU, and (when macmon
# is present) an energy sample stream are written to the results directory.
#
# Each run's pytest session takes the machine-wide GPU lease for as long as it
# lives (tests/conftest.py, tests/gpu_lease.py), so a run waits for any other
# lease holder and the lease is free again between runs. This script does not
# hold it itself; a wrapper that does must export PROXY_GPU_LEASE_HELD=1.
# A run's wall time and its failure record start when it got the lease, and
# the wait before that is recorded on its own (lease_wait_s in results.jsonl).
#
# Usage: scripts/buckshot_gate.sh [N] [results_dir]
set -uo pipefail

N="${1:-5}"
OUT_DIR="${2:-buckshot_gate_results}"
mkdir -p "$OUT_DIR"

# Under the full volley a single request can legitimately queue for 300s+;
# size the per-request HTTP timeout above that so the client cap cannot
# masquerade as an engine failure (tests/functional/cases/_timeout.py).
export ORCHARD_TEST_HTTP_TIMEOUT_S="${ORCHARD_TEST_HTTP_TIMEOUT_S:-600}"

# Who else is on the GPU. The firmware lockup that turns this gate red is
# device-wide: GPU work in any other process (another engine, a PAL pytest
# session, a benchmark) can push the device over the line, and once the run is
# over nothing else says who was there. The GPU lease keeps cooperating
# sessions apart; this names whoever ran beside us anyway.
record_tenants() { # $1 = file to write
  {
    echo "== $(date '+%Y-%m-%dT%H:%M:%S') other engine, python and pytest processes (pid, ppid, started, command)"
    # Everything under this script (its pytest, that pytest's engine) and
    # whatever launched it is the run's own and is left out. An engine listed
    # here with a start time inside the run is one a failed run left behind.
    ps -axww -o pid=,ppid=,lstart=,command= | awk -v self="$$" '
      { pid[NR] = $1; parent[$1] = $2; line[NR] = $0 }
      END {
        for (p = self; p > 1; p = parent[p]) launcher[p] = 1
        for (i = 1; i <= NR; i++) {
          own = pid[i] in launcher
          for (p = pid[i]; p > 1; p = parent[p]) if (p == self) { own = 1; break }
          if (!own && tolower(line[i]) ~ /proxy_inference_engine|python|pytest/) print line[i]
        }
      }'
    echo "== device-wide GPU memory in use, bytes (IOAccelerator PerformanceStatistics)"
    ioreg -r -d 1 -w 0 -c IOAccelerator | grep -o '"In use system memory"=[0-9]*'
  } > "$1" 2>&1
}

# The GPU error callbacks the system logged during the run, grouped by process.
# Every lockup recorded so far errored exactly 19 command buffers device-wide,
# so fewer than 19 under our engine's pid means another process held the rest:
# any other pid in this list was a GPU tenant at the moment of the lockup, even
# if it has exited by now. The window is the whole time the run held the GPU
# lease, not the last few minutes: a hung suite is only declared after its
# 480 s timeout, so the lockup behind a red run can be that old. It does not
# reach back into the wait for the lease: GPU errors from then belong to the
# session the run queued behind. (`log` records its own command line, which
# contains the string we search for; the predicate leaves that out.)
record_gpu_errors() { # $1 = file to append to, $2 = window start "YYYY-MM-DD HH:MM:SS"
  {
    echo "== GPU error callbacks since $2: count, process[pid], kind, first seen"
    /usr/bin/log show --start "$2" --style compact \
      --predicate 'eventMessage CONTAINS "kIOGPUCommandBufferCallbackError" AND process != "log"' |
      awk '
        match($0, /kIOGPUCommandBufferCallbackError[A-Za-z]*/) {
          kind = substr($0, RSTART, RLENGTH)
          who = match($0, /[^ ]+\[[0-9]+:/) ? substr($0, RSTART, RLENGTH - 1) "]" : "?"
          key = who " " kind
          if (!(key in n)) { first[key] = $1 " " $2; order[++keys] = key }
          n[key]++; total++; raw[total] = $0
        }
        END {
          for (k = 1; k <= keys; k++) print n[order[k]], order[k], first[order[k]]
          print total + 0, "errored command buffers in total"
          for (t = 1; t <= total; t++) print "  " raw[t]
        }'
  } >> "$1" 2>&1
}

for i in $(seq 1 "$N"); do
  log="$OUT_DIR/run_${i}.log"
  # Keep each run's engine log (and any hang sample) next to its results:
  # tests/conftest.py wipes its log directory before it starts the engine, so
  # without a per-run directory run N+1 erases the engine-side record of run N,
  # and the workflow uploads only the results directory.
  export ORCHARD_TEST_LOG_DIR="$(cd "$OUT_DIR" && pwd)/run_${i}_logs"
  # Taken before pytest starts, so before any wait for the GPU lease: a session
  # listed here may be the holder this run then waited for (the waiting line
  # is in run_N.log), not one that ran beside it.
  run_started="$(date '+%Y-%m-%d %H:%M:%S')"
  record_tenants "$OUT_DIR/run_${i}_tenants_start.txt"
  macmon_pid=""
  if command -v macmon >/dev/null 2>&1; then
    macmon pipe -i 1000 > "$OUT_DIR/run_${i}_energy.jsonl" 2>/dev/null &
    macmon_pid=$!
  fi

  start=$(date +%s)
  python -m pytest -m buckshot -q -s 2>&1 | tee "$log"
  status=${PIPESTATUS[0]}
  ended=$(date +%s)
  wall=$(( ended - start ))

  # The run may have spent most of that waiting for the GPU lease, and what the
  # GPU did meanwhile is the lease holder's. The session prints "[gpu-lease]
  # acquired <time>" once it holds the lease (tests/conftest.py); the wall time
  # and the failure record count from there. lease_wait_s runs from launching
  # pytest to that line, so it is a few seconds of startup when nobody held
  # the lease. Without the line (the run took no lease: a wrapper holds it, or
  # PROXY_GPU_LEASE=0; or it died first) everything counts from the run's start.
  held_from="$(grep -m 1 -o '\[gpu-lease\] acquired [0-9-]\{10\} [0-9:]\{8\}' "$log" | cut -d ' ' -f 3-)"
  lease_wait=null
  if [ -n "$held_from" ] && held_from_s=$(date -j -f '%Y-%m-%d %H:%M:%S' "$held_from" +%s 2>/dev/null); then
    lease_wait=$(( held_from_s - start ))
    wall=$(( ended - held_from_s ))
  fi
  held_from="${held_from:-$run_started}"

  if [ -n "$macmon_pid" ]; then
    kill "$macmon_pid" 2>/dev/null || true
    # Reap it here, or bash reports the kill into whatever is written next.
    wait "$macmon_pid" 2>/dev/null || true
  fi

  if [ "$status" -ne 0 ]; then
    record_tenants "$OUT_DIR/run_${i}_tenants_failure.txt"
    record_gpu_errors "$OUT_DIR/run_${i}_tenants_failure.txt" "$held_from"
    # The system's own report of each GPU restart while the run held the lease.
    mkdir -p "$ORCHARD_TEST_LOG_DIR"
    find /Library/Logs/DiagnosticReports -maxdepth 1 -name 'gpuEvent-*' \
      -newermt "$held_from" \
      -exec cp {} "$ORCHARD_TEST_LOG_DIR/" \; 2>/dev/null || true
  fi

  python - "$OUT_DIR/results.jsonl" "$i" "$wall" "$status" "$log" \
    "$ORCHARD_TEST_LOG_DIR/engine.test.log" "$lease_wait" <<'PY'
import json, re, sys

out, run, wall, status, log, engine_log, lease_wait = sys.argv[1:]
suites = []
for line in open(log):
    m = re.match(r"\[buckshot\] (\S+)\s+(\S+)\s+([\d.]+)s\s+(.*)", line.strip())
    if m:
        suites.append(
            {"suite": m[1], "name": m[2], "secs": float(m[3]), "state": m[4].strip()}
        )
longest = max((s["secs"] for s in suites), default=0.0)
# A GPU-wide freeze shows in the engine log as command buffer acquisition
# stalling on the busy streams at once. The firmware's restart step is about
# half a second, and the same stall has been seen to resolve with no restart,
# so waits of 400 ms and up are counted for every run, green or red: the count
# is what to compare between runs that had the GPU alone and runs that did not.
freeze_waits = None
try:
    with open(engine_log, errors="replace") as f:
        waits = re.findall(
            r"command buffer acquisition on stream \d+ waited ([\d.]+) ms", f.read()
        )
    freeze_waits = sum(float(ms) >= 400 for ms in waits)
except FileNotFoundError:
    pass
shown = "no engine log" if freeze_waits is None else freeze_waits
print(f"GATE run {run}: GPU-wide freezes (acquisition waits >= 400 ms): {shown}")
if lease_wait != "null":
    print(f"GATE run {run}: held the GPU lease {lease_wait}s after launch")
record = {
    "run": int(run),
    "lease_wait_s": json.loads(lease_wait),
    "wall_s": int(wall),
    "longest_suite_s": longest,
    "stopwatch_gap_s": int(wall) - longest,
    "passed": status == "0",
    "gpu_freeze_waits": freeze_waits,
    "suites": suites,
}
with open(out, "a") as f:
    f.write(json.dumps(record) + "\n")
PY

  if [ "$status" -ne 0 ]; then
    echo "GATE FAILED at run ${i}/${N}"
    exit 1
  fi
  echo "GATE run ${i}/${N} green (${wall}s)"
done

echo "GATE PASSED: ${N} consecutive green runs"
