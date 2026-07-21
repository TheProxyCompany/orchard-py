#!/usr/bin/env bash
# Buckshot N-consecutive-runs gate.
#
# Runs the full buckshot matrix N times, each in a fresh pytest session (fresh
# engine process, full hydration), aborting on the first red. A single failure
# resets the streak by definition — rerun the whole gate. Per-run wall clock,
# per-suite timings, and (when macmon is present) an energy sample stream are
# written to the results directory.
#
# Usage: scripts/buckshot_gate.sh [N] [results_dir]
set -uo pipefail

N="${1:-5}"
OUT_DIR="${2:-buckshot_gate_results}"
mkdir -p "$OUT_DIR"

for i in $(seq 1 "$N"); do
  log="$OUT_DIR/run_${i}.log"
  macmon_pid=""
  if command -v macmon >/dev/null 2>&1; then
    macmon pipe -i 1000 > "$OUT_DIR/run_${i}_energy.jsonl" 2>/dev/null &
    macmon_pid=$!
  fi

  start=$(date +%s)
  python -m pytest -m buckshot -q -s 2>&1 | tee "$log"
  status=${PIPESTATUS[0]}
  wall=$(( $(date +%s) - start ))

  if [ -n "$macmon_pid" ]; then
    kill "$macmon_pid" 2>/dev/null || true
  fi

  python - "$OUT_DIR/results.jsonl" "$i" "$wall" "$status" "$log" <<'PY'
import json, re, sys

out, run, wall, status, log = sys.argv[1:]
suites = []
for line in open(log):
    m = re.match(r"\[buckshot\] (\S+)\s+(\S+)\s+([\d.]+)s\s+(.*)", line.strip())
    if m:
        suites.append(
            {"suite": m[1], "name": m[2], "secs": float(m[3]), "state": m[4].strip()}
        )
longest = max((s["secs"] for s in suites), default=0.0)
record = {
    "run": int(run),
    "wall_s": int(wall),
    "longest_suite_s": longest,
    "stopwatch_gap_s": int(wall) - longest,
    "passed": status == "0",
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
