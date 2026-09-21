"""Buckshot: every model's functional + golden suite in one concurrent volley.

The engine fixture preloads the full chat matrix; the pipeline models are
loaded on top, one at a time, before the first request. Then every suite is
fired at the engine at once and the engine's own scheduling handles the load.
Nothing is throttled, ordered, or excluded on the client side.

Opt-in (it duplicates the per-model matrices): python -m pytest -m buckshot -q -s
"""

import asyncio
import shutil
import subprocess
import time

import pytest

from orchard.engine.io import get_engine_file_paths
from orchard.engine.multiprocess import pid_is_alive, read_pid_file
from tests.conftest import LOG_DIR
from tests.functional.cases.registry import cases_for_model as functional_cases
from tests.functional.cases.registry import run_cases as run_functional
from tests.golden import golden_io
from tests.golden.cases.registry import model_cases, pipeline_cases
from tests.golden.cases.registry import run_cases as run_golden
from tests.models import MODELS, PIPELINE_TOOL_MODELS, Model

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.buckshot,
]

SUITE_TIMEOUT_S = 480


def _snapshot_hung_engine() -> str:
    """Record where the engine's threads are while a suite is still hung.

    A suite that times out while the rest of the volley completes is a wedged
    scheduler inside a live engine (a lock cycle, a lost GPU completion). The
    engine log does not show that; a stack sample does, and it has to be taken
    here, before teardown stops the engine.
    """
    pid = read_pid_file(get_engine_file_paths(None, None).pid_file)
    if pid is None or not pid_is_alive(pid):
        return f"engine pid {pid} is not alive"
    sampler = shutil.which("sample")
    if sampler is None:
        return f"engine pid {pid} is alive (no `sample` tool to capture its stacks)"
    out = LOG_DIR / "engine_hang.sample.txt"
    subprocess.run(
        [sampler, str(pid), "2", "-mayDie", "-file", str(out)],
        capture_output=True,
        timeout=120,
        check=False,
    )
    return f"engine pid {pid} is alive; thread stacks in {out}"


async def test_buckshot_full_matrix(live_server, client, engine):
    fixtures = {"live_server": live_server, "client": client, "engine": engine}

    async def run_suite(suite, name, coro_factory):
        start = time.perf_counter()
        try:
            failures = await asyncio.wait_for(coro_factory(), SUITE_TIMEOUT_S)
            timed_out = False
        except TimeoutError:
            failures, timed_out = [], True
        secs = time.perf_counter() - start
        state = "TIMEOUT" if timed_out else f"{len(failures)} fail"
        print(f"[buckshot] {suite:11s} {name:15s} {secs:6.1f}s  {state}", flush=True)
        return (suite, name, secs, failures, timed_out)

    async def functional_suite(model: Model):
        failures, _skipped = await run_functional(
            functional_cases(model), fixtures, model
        )
        return failures

    # Loading is part of turning the engine on, not part of the volley:
    # hydrating all the modal models at once on top of the resident chat
    # matrix spikes wired memory past the Metal limit.
    for model_id in PIPELINE_TOOL_MODELS:
        await engine.load_models([model_id])

    jobs = [
        run_suite(
            "golden",
            "pipeline",
            lambda: run_golden(pipeline_cases(), {"client": client}),
        )
    ]
    for m in MODELS:
        jobs.append(
            run_suite("functional", m.template_type, lambda m=m: functional_suite(m))
        )
        jobs.append(
            run_suite(
                "golden",
                m.template_type,
                lambda m=m: run_golden(model_cases(), {"client": client}, m),
            )
        )

    wall_start = time.perf_counter()
    results = await asyncio.gather(*jobs)
    wall = time.perf_counter() - wall_start

    print(f"\nBUCKSHOT wall: {wall:.1f}s ({len(results)} suites)")
    print(f"{'suite':11s} {'model':15s} {'secs':>6s}  result")
    for suite, name, secs, failures, timed_out in sorted(results, key=lambda r: -r[2]):
        state = "TIMEOUT" if timed_out else f"{len(failures)} fail"
        print(f"{suite:11s} {name:15s} {secs:6.1f}  {state}")
        for failure in failures:
            tail = failure.detail.strip().splitlines()[-4:]
            print(f"    x {failure.case_id}:")
            for line in tail:
                print(f"      | {line[:400]}")

    hung = [(s, n, round(t)) for s, n, t, _, timed_out in results if timed_out]
    if hung:
        engine_state = await asyncio.to_thread(_snapshot_hung_engine)
        print(f"BUCKSHOT hang: {engine_state}", flush=True)
        pytest.fail(f"suites timed out at {SUITE_TIMEOUT_S}s: {hung}; {engine_state}")

    failed = [
        (s, n, [f.case_id for f in failures])
        for s, n, _, failures, _ in results
        if failures
    ]
    assert not failed, f"suites had case failures: {failed}"

    # Buckshot bypasses the tests/golden conftest record-on-pass hooks, so a
    # missing baseline would otherwise stage silently and "pass". Missing
    # goldens are a failure here; record them via the golden matrix.
    missing = [str(path) for path in golden_io.pending_paths()]
    golden_io.discard_pending()
    assert not missing, (
        f"golden baselines missing (record via golden matrix): {missing}"
    )
