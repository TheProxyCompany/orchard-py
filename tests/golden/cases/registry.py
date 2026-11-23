from __future__ import annotations

import asyncio
import inspect
import time
import traceback
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import pytest

from tests.models import Model

from . import (
    audio_telephone,
    image_tool_result_grounding,
    multi_tool,
    reason_then_structured,
    reason_then_tool,
    thinking_on_off,
    tool_chaining,
    tool_result_grounding,
    tool_selection,
)


_MODEL_MODULES = [
    multi_tool,
    reason_then_structured,
    reason_then_tool,
    thinking_on_off,
    tool_chaining,
    tool_result_grounding,
    tool_selection,
]

_PIPELINE_MODULES = [
    audio_telephone,
    image_tool_result_grounding,
]


@dataclass(frozen=True)
class GoldenCase:
    id: str
    function: Callable[..., Any]

    async def run(self, fixtures: dict[str, Any], model: Model | None = None) -> None:
        kwargs: dict[str, Any] = {}
        for name in inspect.signature(self.function).parameters:
            if name == "model":
                kwargs[name] = model
            else:
                kwargs[name] = fixtures[name]

        if inspect.iscoroutinefunction(self.function):
            await self.function(**kwargs)
            return
        await asyncio.to_thread(self.function, **kwargs)


@dataclass(frozen=True)
class CaseFailure:
    case_id: str
    detail: str


def model_cases() -> list[GoldenCase]:
    return _collect(_MODEL_MODULES)


def pipeline_cases(family: str | None = None) -> list[GoldenCase]:
    """Pipeline golden cases; ``family`` = "audio" or "image" narrows to one
    modal family."""
    modules = {
        None: _PIPELINE_MODULES,
        "audio": [audio_telephone],
        "image": [image_tool_result_grounding],
    }[family]
    return _collect(modules)


async def run_cases(
    cases: Iterable[GoldenCase],
    fixtures: dict[str, Any],
    model: Model | None = None,
) -> list[CaseFailure]:
    label = model.template_type if model else "pipeline"

    async def run_one(case: GoldenCase) -> tuple[str, BaseException | None]:
        # One line per finished case: a suite cut off at its wall says only
        # how long the wall was, and these say which cases it never reached.
        start = time.perf_counter()
        exc: BaseException | None = None
        try:
            await case.run(fixtures, model)
        except (pytest.skip.Exception, pytest.fail.Exception, Exception) as caught:
            exc = caught
        secs = time.perf_counter() - start
        state = "ok" if exc is None else type(exc).__name__
        print(f"[case] golden {label} {case.id} {secs:.1f}s {state}", flush=True)
        return case.id, exc

    results = await asyncio.gather(*(run_one(case) for case in cases))
    failures: list[CaseFailure] = []
    for case_id, exc in results:
        if exc is None:
            continue
        if isinstance(exc, pytest.skip.Exception):
            continue
        detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        failures.append(CaseFailure(case_id, detail))
    return failures


def format_failures(label: str, failures: list[CaseFailure]) -> str:
    parts = [f"{label} golden matrix failed {len(failures)} case(s):"]
    for failure in failures:
        parts.append(f"\n--- {failure.case_id} ---\n{failure.detail}")
    return "\n".join(parts)


def _collect(modules: Iterable[Any]) -> list[GoldenCase]:
    cases: list[GoldenCase] = []
    for module in modules:
        scenario = module.__name__.rsplit(".", 1)[-1]
        for name, function in sorted(vars(module).items()):
            if name.startswith("test_") and callable(function):
                cases.append(GoldenCase(f"{scenario}.{name.removeprefix('test_')}", function))
    return cases
