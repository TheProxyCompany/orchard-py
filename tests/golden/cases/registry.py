from __future__ import annotations

import asyncio
import inspect
import traceback
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, ClassVar

import pytest

from tests.golden.golden_io import collect_drift
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


class GreedyClient:
    """The client every golden case gets: its Responses calls decode greedily.
    Why, and what passes through untouched: tests/golden/README.md, "Sampling"."""

    SAMPLING: ClassVar[dict[str, Any]] = {"temperature": 0.0, "deterministic": True}

    def __init__(self, client: Any) -> None:
        self._client = client

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    async def aresponses(self, model_id: str, **kwargs: Any) -> Any:
        return await self._client.aresponses(model_id, **{**kwargs, **self.SAMPLING})

    async def arender_responses_prompt(self, model_id: str, **kwargs: Any) -> Any:
        return await self._client.arender_responses_prompt(
            model_id, **{**kwargs, **self.SAMPLING}
        )


@dataclass(frozen=True)
class GoldenCase:
    id: str
    function: Callable[..., Any]

    async def run(self, fixtures: dict[str, Any], model: Model | None = None) -> None:
        kwargs: dict[str, Any] = {}
        for name in inspect.signature(self.function).parameters:
            if name == "model":
                kwargs[name] = model
            elif name == "client":
                kwargs[name] = GreedyClient(fixtures[name])
            else:
                kwargs[name] = fixtures[name]

        # Every turn of the case is compared, and drifted turns fail it at the end.
        with collect_drift():
            if inspect.iscoroutinefunction(self.function):
                await self.function(**kwargs)
            else:
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
    async def run_one(case: GoldenCase) -> tuple[str, BaseException | None]:
        try:
            await case.run(fixtures, model)
        except pytest.skip.Exception as exc:
            return case.id, exc
        except Exception as exc:
            return case.id, exc
        return case.id, None

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
                cases.append(
                    GoldenCase(f"{scenario}.{name.removeprefix('test_')}", function)
                )
    return cases
