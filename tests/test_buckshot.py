"""Buckshot: every model's functional + golden suite in one concurrent volley.

The engine fixture already preloads the full matrix; the per-model suites
already fire their cases in one asyncio.gather. This collapses the remaining
serial axis — models and suites — into the same gather, so the whole matrix
becomes one continuous batch against the engine.

Opt-in (it duplicates the per-model matrices): BUCKSHOT=1 python -m pytest
tests/test_buckshot.py -q -s
BUCKSHOT_WIDTH=N caps concurrent suites (0 = unbounded).
BUCKSHOT_SKIP=a,b excludes models by template_type.
"""

import asyncio
import os
import time

import pytest

from tests.functional.cases.registry import cases_for_model as functional_cases
from tests.functional.cases.registry import run_cases as run_functional
from tests.golden.cases.registry import model_cases, pipeline_cases
from tests.golden.cases.registry import run_cases as run_golden
from tests.models import MODELS, Model

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not os.getenv("BUCKSHOT"), reason="opt-in: set BUCKSHOT=1"),
]

SUITE_TIMEOUT_S = 480

# The pipeline suite's tool models activate on demand; under a full buckshot
# that means hydrating a diffusion model while the GPU is saturated, which
# trips the macOS watchdog and kills the engine. Activate them up front on an
# idle GPU instead, like the chat matrix.
PIPELINE_TOOL_MODELS = [
    "ideogram-ai/ideogram-4-fp8",
    "black-forest-labs/FLUX.2-klein-4B",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "mlx-community/parakeet-tdt-0.6b-v3",
    "Qwen/Qwen3-ASR-0.6B",
    "Qwen/Qwen3-ASR-1.7B",
]


async def test_buckshot_full_matrix(live_server, client, engine):
    fixtures = {"live_server": live_server, "client": client, "engine": engine}
    # Unbounded width (21 suites, ~10 concurrent model schedulers) trips the
    # macOS GPU watchdog: one command buffer hangs, recovery kills the rest.
    # Cap concurrent suites until the engine paces command buffers itself.
    width = int(os.getenv("BUCKSHOT_WIDTH", "4"))
    gate = asyncio.Semaphore(width if width > 0 else len(MODELS) * 2 + 1)
    skip = {s for s in os.getenv("BUCKSHOT_SKIP", "").split(",") if s}
    models = [m for m in MODELS if m.template_type not in skip]

    async def run_suite(suite, name, coro_factory):
        async with gate:
            start = time.perf_counter()
            try:
                failures = await asyncio.wait_for(coro_factory(), SUITE_TIMEOUT_S)
                timed_out = False
            except TimeoutError:
                failures, timed_out = [], True
            secs = time.perf_counter() - start
            state = "TIMEOUT" if timed_out else f"{len(failures)} fail"
            print(f"[buckshot] {suite:11s} {name:15s} {secs:6.1f}s  {state}",
                  flush=True)
            return (suite, name, secs, failures, timed_out)

    async def functional_suite(model: Model):
        failures, skipped = await run_functional(
            functional_cases(model), fixtures, model
        )
        return failures

    jobs = [
        run_suite("functional", m.template_type,
                  lambda m=m: functional_suite(m))
        for m in models
    ] + [
        run_suite("golden", m.template_type,
                  lambda m=m: run_golden(model_cases(), {"client": client}, m))
        for m in models
    ]
    if "pipeline" not in skip:
        await engine.load_models(PIPELINE_TOOL_MODELS)
        jobs.append(run_suite("golden", "pipeline",
                              lambda: run_golden(pipeline_cases(), {"client": client})))

    wall_start = time.perf_counter()
    results = await asyncio.gather(*jobs)
    wall = time.perf_counter() - wall_start

    print(f"\nBUCKSHOT wall: {wall:.1f}s "
          f"({len(results)} suites, width={width}, skip={sorted(skip) or 'none'})")
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
    assert not hung, f"suites timed out at {SUITE_TIMEOUT_S}s: {hung}"

    failed = [
        (s, n, [f.case_id for f in failures])
        for s, n, _, failures, _ in results
        if failures
    ]
    assert not failed, f"suites had case failures: {failed}"
