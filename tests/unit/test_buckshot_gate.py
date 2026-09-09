from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
GATE = REPO_ROOT / "scripts" / "buckshot_gate.sh"


@pytest.fixture
def gate_environment(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "if sys.argv[1:3] == ['-m', 'pytest']:\n"
        "    with open(os.environ['GATE_TRACE'], 'a') as trace:\n"
        "        trace.write(json.dumps({'cwd': os.getcwd(), 'args': sys.argv[1:], "
        "'http_timeout': os.environ.get('ORCHARD_TEST_HTTP_TIMEOUT_S')}) + '\\n')\n"
        "    print('[buckshot] functional synthetic 1.0s 0 fail')\n"
        "    sys.exit(int(os.environ.get('FAKE_PYTEST_STATUS', '0')))\n"
        "os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
    )
    python.chmod(0o755)
    macmon = bin_dir / "macmon"
    macmon.write_text("#!/bin/sh\nexit 0\n")
    macmon.chmod(0o755)
    env = dict(os.environ)
    env.pop("ORCHARD_TEST_HTTP_TIMEOUT_S", None)
    env.update(PATH=f"{bin_dir}:/usr/bin:/bin", GATE_TRACE=str(tmp_path / "trace.jsonl"))
    return env


def run_gate(tmp_path, env, *args):
    return subprocess.run(
        ["/bin/bash", str(GATE), *args],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )


def attempts(env):
    path = Path(env["GATE_TRACE"])
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_gate_runs_its_own_checkout_from_another_working_directory(tmp_path, gate_environment):
    result = run_gate(tmp_path, gate_environment, "1", "results")
    assert result.returncode == 0, result.stderr
    assert attempts(gate_environment)[0]["cwd"] == str(REPO_ROOT)
    assert (tmp_path / "results" / "results.jsonl").exists()


@pytest.mark.parametrize("count", ["0", "-1", "not-a-count", "1.5"])
def test_gate_rejects_invalid_session_count_without_reporting_green(tmp_path, gate_environment, count):
    result = run_gate(tmp_path, gate_environment, count, "results")
    assert result.returncode != 0
    assert "GATE PASSED" not in result.stdout
    assert not attempts(gate_environment)


def test_gate_defaults_to_five_full_sessions(tmp_path, gate_environment):
    result = run_gate(tmp_path, gate_environment)
    assert result.returncode == 0, result.stderr
    observed = attempts(gate_environment)
    assert len(observed) == 5
    assert all(row["args"] == ["-m", "pytest", "-m", "buckshot", "-q", "-s"] for row in observed)
    assert all(row["http_timeout"] == "600" for row in observed)


def test_gate_stops_at_first_failed_session(tmp_path, gate_environment):
    gate_environment["FAKE_PYTEST_STATUS"] = "1"
    result = run_gate(tmp_path, gate_environment, "5", "results")
    assert result.returncode != 0
    assert len(attempts(gate_environment)) == 1
    assert "GATE PASSED" not in result.stdout
    record = json.loads((tmp_path / "results" / "results.jsonl").read_text())
    assert record["passed"] is False
