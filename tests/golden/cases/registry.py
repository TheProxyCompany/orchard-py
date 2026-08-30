from __future__ import annotations

import asyncio
import inspect
import traceback
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Protocol

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

_REQUIRES_TOOLS = {
    multi_tool.test_multi_tool,
    reason_then_tool.test_reason_then_tool,
    tool_chaining.test_tool_chaining,
    tool_result_grounding.test_tool_result_grounding,
    tool_selection.test_tool_selection,
}
_REQUIRES_THINKING = {reason_then_structured.test_reason_then_structured}
_REQUIRES_TOGGLEABLE_THINKING = {thinking_on_off.test_thinking_on_off}

_REQUEST_PHASES = {
    audio_telephone.test_tts_to_speech_to_text_transcription: (12, 36),
    image_tool_result_grounding.test_image_tool_self_loop_and_blind_verifier: (
        1,
        1,
        2,
    ),
    image_tool_result_grounding.test_image_tool_self_loop_and_blind_verifier_flux: (
        1,
        1,
        2,
    ),
    image_tool_result_grounding.test_image_edit_tool_blind_verifier: (1, 1, 1, 1),
    image_tool_result_grounding.test_image_edit_tool_blind_verifier_flux: (
        1,
        1,
        1,
        1,
    ),
    multi_tool.test_multi_tool: (1, 1, 1),
    reason_then_tool.test_reason_then_tool: (1, 1),
    thinking_on_off.test_thinking_on_off: (2,),
    tool_chaining.test_tool_chaining: (1, 1, 1),
    tool_result_grounding.test_tool_result_grounding: (1, 1),
}


@dataclass(frozen=True)
class GoldenCase:
    id: str
    function: Callable[..., Any]

    @property
    def request_phases(self) -> tuple[int, ...]:
        return _REQUEST_PHASES.get(self.function, (1,))

    def applies_to(self, model: Model) -> bool:
        if self.function in _REQUIRES_TOOLS:
            return model.tools
        if self.function in _REQUIRES_THINKING:
            return bool(model.thinking)
        if self.function in _REQUIRES_TOGGLEABLE_THINKING:
            return model.thinking is True
        return True

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


class CaseAdmission(Protocol):
    async def admit(self, case_id: str) -> None: ...

    def complete(self, case_id: str, *, skipped: bool = False) -> None: ...


def model_cases(model: Model) -> list[GoldenCase]:
    return [case for case in _collect(_MODEL_MODULES) if case.applies_to(model)]


def pipeline_cases() -> list[GoldenCase]:
    return _collect(_PIPELINE_MODULES)


async def run_cases(
    cases: Iterable[GoldenCase],
    fixtures: dict[str, Any],
    model: Model | None = None,
    *,
    admission: CaseAdmission | None = None,
    case_prefix: str = "",
) -> list[CaseFailure]:
    async def run_one(case: GoldenCase) -> tuple[str, BaseException | None]:
        admitted_id = f"{case_prefix}{case.id}"
        skipped = False
        if admission is not None:
            await admission.admit(admitted_id)
        try:
            await case.run(fixtures, model)
        except pytest.skip.Exception as exc:
            skipped = True
            return case.id, exc
        except Exception as exc:  # noqa: BLE001 - aggregate every case failure
            return case.id, exc
        finally:
            if admission is not None:
                admission.complete(admitted_id, skipped=skipped)
        return case.id, None

    results = await asyncio.gather(*(run_one(case) for case in cases))
    failures: list[CaseFailure] = []
    for case_id, exc in results:
        if exc is None:
            continue
        if isinstance(exc, pytest.skip.Exception):
            if admission is not None:
                failures.append(
                    CaseFailure(case_id, f"Buckshot forbids skipped cases: {exc}")
                )
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
