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
    # Width >= 4 (let alone unbounded, 21 suites / ~10 concurrent model
    # schedulers) trips the macOS GPU watchdog: one command buffer hangs,
    # recovery kills the rest of the engine. Width 2 is the proven-stable
    # cap until the engine paces command buffers itself.
    width = int(os.getenv("BUCKSHOT_WIDTH", "2"))
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
            print(
                f"[buckshot] {suite:11s} {name:15s} {secs:6.1f}s  {state}", flush=True
            )
            return (suite, name, secs, failures, timed_out)

    async def functional_suite(model: Model):
        failures, _skipped = await run_functional(
            functional_cases(model), fixtures, model
        )
        return failures

    # Keep giants apart: the semaphore serves jobs in creation order, and two
    # giants sharing the width-2 window starve each other (measured twice:
    # nemotron_h golden 45s solo -> 254s beside the pipeline suite at the
    # tail, -> 241s beside it at the head). The pipeline suite goes first and
    # owns one slot for ~230s; light suites cycle through the other slot;
    # heavy chat suites run LAST, after the pipeline suite is done or nearly
    # done, ordered so nemotron_h — the proven starvation victim — enters
    # when the GPU is quietest.
    TAIL = [
        ("golden", "lfm2_5"),
        ("golden", "afmoe"),
        ("functional", "lfm2_5"),
        ("golden", "nemotron_h"),
    ]

    def suite_jobs():
        entries = []
        for m in models:
            entries.append(
                ("functional", m.template_type, lambda m=m: functional_suite(m))
            )
            entries.append(
                (
                    "golden",
                    m.template_type,
                    lambda m=m: run_golden(model_cases(), {"client": client}, m),
                )
            )
        entries.sort(
            key=lambda e: (TAIL.index((e[0], e[1])) if (e[0], e[1]) in TAIL else -1)
        )
        return entries

    wall_start = time.perf_counter()
    results = []
    if "pipeline" not in skip:
        # Preload MUST happen on an idle GPU, before the volley: activating
        # diffusion/TTS models while chat decode is in flight trips the GPU
        # watchdog and kills the engine (re-confirmed 2026-07-08 when this
        # load was briefly moved inside the volley). Load them one at a time:
        # hydrating all seven concurrently on top of the resident chat matrix
        # spikes wired memory past the Metal limit and the engine dies with a
        # silent abort (three full-matrix runs, 2026-07-10).
        for model_id in PIPELINE_TOOL_MODELS:
            await engine.load_models([model_id])
        # The diffusion window runs the GPU exclusively. Chat decode beside
        # active diffusion starves 3-10x (measured: nemotron_h golden 45s ->
        # 241-254s, gemma4 golden 13s -> 155s), pushing first-token latency
        # past the server's delta timeout — whole suites 502. Every type-1
        # GPU-restart engine death on record also had diffusion + chat
        # in flight together. Until the engine paces GPU work across model
        # runtimes, the pipeline suite and the chat volley don't overlap.
        results.append(
            await run_suite(
                "golden",
                "pipeline",
                lambda: run_golden(pipeline_cases(), {"client": client}),
            )
        )
    jobs = [run_suite(suite, name, factory) for suite, name, factory in suite_jobs()]
    results += await asyncio.gather(*jobs)
    wall = time.perf_counter() - wall_start

    print(
        f"\nBUCKSHOT wall: {wall:.1f}s "
        f"({len(results)} suites, width={width}, skip={sorted(skip) or 'none'})"
    )
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
