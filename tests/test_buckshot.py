"""Fire every applicable functional and golden case in one admitted volley."""

import asyncio
import time

import pytest

from orchard.app.ipc_dispatch import IPCState
from tests.functional.cases.registry import CaseFailure
from tests.functional.cases.registry import cases_for_model as functional_cases
from tests.functional.cases.registry import run_cases as run_functional
from tests.golden import golden_io
from tests.golden.cases.registry import model_cases, pipeline_cases
from tests.golden.cases.registry import run_cases as run_golden
from tests.models import MODELS, PIPELINE_TOOL_MODELS, Model
from tests.shared_owner import (
    BuckshotAdmissionGate,
    observe_request_attempt,
    shared_owner_enabled,
)

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.buckshot,
]


async def test_buckshot_full_matrix(live_server, client, engine, monkeypatch):
    fixtures = {"live_server": live_server, "client": client, "engine": engine}
    models = MODELS
    functional_population = {
        model.template_type: functional_cases(model) for model in models
    }
    golden_population = {model.template_type: model_cases(model) for model in models}
    pipeline_population = pipeline_cases()
    request_plan = {
        **{
            f"functional::{model.template_type}::{case.id}": case.request_phases
            for model in models
            for case in functional_population[model.template_type]
        },
        **{
            f"golden::{model.template_type}::{case.id}": case.request_phases
            for model in models
            for case in golden_population[model.template_type]
        },
        **{
            f"golden::pipeline::{case.id}": case.request_phases
            for case in pipeline_population
        },
    }
    model_ids = [
        *dict.fromkeys([*(model.checkpoint for model in models), *PIPELINE_TOOL_MODELS])
    ]
    admission = BuckshotAdmissionGate(model_ids, request_plan)

    original_send_request = IPCState.send_request

    async def observed_send_request(self, request_bytes):
        await observe_request_attempt()
        await original_send_request(self, request_bytes)

    monkeypatch.setattr(IPCState, "send_request", observed_send_request)

    async def run_suite(suite, name, coro_factory):
        start = time.perf_counter()
        failures = await coro_factory()
        secs = time.perf_counter() - start
        print(
            f"[buckshot] {suite:11s} {name:15s} {secs:6.1f}s  {len(failures)} fail",
            flush=True,
        )
        return (suite, name, secs, failures)

    async def functional_suite(model: Model):
        failures, skipped = await run_functional(
            functional_population[model.template_type],
            fixtures,
            model,
            admission=admission,
            case_prefix=f"functional::{model.template_type}::",
        )
        for skipped_case in skipped:
            case_id, _, detail = skipped_case.partition(": ")
            failures.append(
                CaseFailure(
                    case_id=case_id,
                    detail=f"Buckshot forbids skipped cases: {detail}",
                )
            )
        return failures

    def suite_jobs():
        entries = []
        for model in models:
            entries.append(
                (
                    "functional",
                    model.template_type,
                    lambda model=model: functional_suite(model),
                )
            )
            entries.append(
                (
                    "golden",
                    model.template_type,
                    lambda model=model: run_golden(
                        golden_population[model.template_type],
                        {"client": client},
                        model,
                        admission=admission,
                        case_prefix=f"golden::{model.template_type}::",
                    ),
                )
            )
        return entries

    wall_start = time.perf_counter()
    pipeline_job = run_suite(
        "golden",
        "pipeline",
        lambda: run_golden(
            pipeline_population,
            {"client": client},
            admission=admission,
            case_prefix="golden::pipeline::",
        ),
    )
    jobs = [run_suite(suite, name, factory) for suite, name, factory in suite_jobs()]
    jobs.insert(0, pipeline_job)
    results = []
    success = False
    try:
        # Every case coroutine reaches admission before any case executes. The
        # final arrival publishes the exact population and waits for the one
        # release signal shared with the Rust process.
        results = await asyncio.gather(*jobs)
        wall = time.perf_counter() - wall_start

        if not shared_owner_enabled():
            print(
                f"\nBUCKSHOT wall: {wall:.1f}s "
                f"({len(results)} suites, launch=uncapped, models=all)"
            )
        print(f"{'suite':11s} {'model':15s} {'secs':>6s}  result")
        for suite, name, secs, failures in sorted(results, key=lambda row: -row[2]):
            print(f"{suite:11s} {name:15s} {secs:6.1f}  {len(failures)} fail")
            for failure in failures:
                tail = failure.detail.strip().splitlines()[-4:]
                print(f"    x {failure.case_id}:")
                for line in tail:
                    print(f"      | {line[:400]}")

        failed = [
            (suite, name, [failure.case_id for failure in failures])
            for suite, name, _, failures in results
            if failures
        ]
        assert not failed, f"suites had case failures: {failed}"

        # Missing baselines cannot silently stage and pass this proof.
        missing = [str(path) for path in golden_io.pending_paths()]
        golden_io.discard_pending()
        assert not missing, (
            f"golden baselines missing (record via golden matrix): {missing}"
        )
        success = True
    finally:
        await admission.finish(success=success)
