from __future__ import annotations

import asyncio
import hashlib
import inspect
import traceback
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import product
from typing import Any, Protocol

import pytest

from tests.models import Model

from . import (
    basic,
    batching,
    best_of,
    capabilities,
    determinism,
    embeddings,
    logprobs,
    multi_candidate,
    multi_token,
    multimodal,
    responses_basic,
    responses_client,
    responses_structured,
    responses_tools,
    stop_sequences,
    structured_generation,
    unicode_payload,
)
from . import (
    client as client_cases,
)

_MODULES = [
    basic,
    batching,
    best_of,
    capabilities,
    client_cases,
    determinism,
    embeddings,
    logprobs,
    multi_candidate,
    multi_token,
    multimodal,
    responses_basic,
    responses_client,
    responses_structured,
    responses_tools,
    stop_sequences,
    structured_generation,
    unicode_payload,
]

_MODEL_ARGUMENTS = {
    "any_model_id",
    "model_id",
    "moondream_model_id",
    "text_model_id",
    "vision_model_id",
}

_PINNED_TEMPLATE_TYPES = {
    responses_client.test_client_responses_tool_calling: {"llama3"},
    responses_client.test_client_responses_tool_result_continuation: {"llama3"},
}

_RUN_IN_THREAD = {
    client_cases.test_client_chat_streaming,
    responses_client.test_client_responses_sync_wrapper,
}

_REQUEST_PHASES = {
    batching.test_chat_completion_batch_length_mismatch_returns_422: (),
    best_of.test_chat_completion_best_of_validation_less_than_n: (),
    best_of.test_chat_completion_best_of_streaming_disallowed: (),
    determinism.test_sequential_request_determinism: (1, 1, 1),
    responses_client.test_client_responses_sync_wrapper: (2,),
    responses_client.test_client_responses_text_helpers: (2,),
    responses_client.test_client_responses_tool_result_continuation: (1, 1),
    responses_tools.test_responses_tool_result_continuation: (1, 1),
}


@dataclass(frozen=True)
class FunctionalCase:
    id: str
    function: Callable[..., Any]
    parameters: dict[str, Any]

    @property
    def argument_names(self) -> set[str]:
        return set(inspect.signature(self.function).parameters)

    @property
    def surface(self) -> str:
        args = self.argument_names
        if "live_server" in args:
            return "server"
        if "client" in args or "engine" in args:
            return "client"
        raise ValueError(f"{self.id} does not declare a runnable test surface")

    @property
    def request_phases(self) -> tuple[int, ...]:
        return _REQUEST_PHASES.get(self.function, (1,))

    def applies_to(self, model: Model) -> bool:
        args = self.argument_names
        pinned_template_types = _PINNED_TEMPLATE_TYPES.get(self.function)
        if pinned_template_types is not None:
            return model.template_type in pinned_template_types
        if "vision_model_id" in args:
            return model.vision
        if "moondream_model_id" in args:
            return model.template_type == "moondream3"
        return True

    async def run(self, fixtures: dict[str, Any], model: Model) -> None:
        kwargs = self._kwargs(fixtures, model)
        if self.function in _RUN_IN_THREAD:
            await asyncio.to_thread(_run_function, self.function, kwargs)
            return
        if inspect.iscoroutinefunction(self.function):
            await self.function(**kwargs)
            return
        await asyncio.to_thread(_run_function, self.function, kwargs)

    def _kwargs(self, fixtures: dict[str, Any], model: Model) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        for name in inspect.signature(self.function).parameters:
            if name in self.parameters:
                kwargs[name] = self.parameters[name]
            elif name in _MODEL_ARGUMENTS:
                kwargs[name] = model.checkpoint
            else:
                kwargs[name] = fixtures[name]
        return kwargs


@dataclass(frozen=True)
class CaseFailure:
    case_id: str
    detail: str


class CaseAdmission(Protocol):
    async def admit(self, case_id: str) -> None: ...

    def complete(self, case_id: str, *, skipped: bool = False) -> None: ...


def collect_functional_cases() -> list[FunctionalCase]:
    cases: list[FunctionalCase] = []
    for module in _MODULES:
        scenario = module.__name__.rsplit(".", 1)[-1]
        for name, function in sorted(vars(module).items()):
            if not name.startswith("test_") or not callable(function):
                continue
            for suffix, parameters in _expand_parameters(function):
                case_id = _case_id(scenario, name, suffix)
                cases.append(FunctionalCase(case_id, function, parameters))
    return cases


def cases_for_model(model: Model) -> list[FunctionalCase]:
    return [case for case in collect_functional_cases() if case.applies_to(model)]


def cases_for_model_and_surface(model: Model, surface: str) -> list[FunctionalCase]:
    return [case for case in cases_for_model(model) if case.surface == surface]


async def run_cases(
    cases: Iterable[FunctionalCase],
    fixtures: dict[str, Any],
    model: Model,
    *,
    admission: CaseAdmission | None = None,
    case_prefix: str = "",
) -> tuple[list[CaseFailure], list[str]]:
    async def run_one(case: FunctionalCase) -> tuple[str, BaseException | None]:
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
    skipped: list[str] = []
    for case_id, exc in results:
        if exc is None:
            continue
        if isinstance(exc, pytest.skip.Exception):
            skipped.append(f"{case_id}: {exc}")
            continue
        detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        failures.append(CaseFailure(case_id, detail))
    return failures, skipped


def format_failures(model: Model, failures: list[CaseFailure]) -> str:
    parts = [f"{model.template_type} functional matrix failed {len(failures)} case(s):"]
    for failure in failures:
        parts.append(f"\n--- {failure.case_id} ---\n{failure.detail}")
    return "\n".join(parts)


def _expand_parameters(
    function: Callable[..., Any],
) -> list[tuple[str, dict[str, Any]]]:
    dimensions: list[tuple[list[str], list[Any]]] = []
    for mark in getattr(function, "pytestmark", ()):
        if mark.name != "parametrize":
            continue
        names = [name.strip() for name in str(mark.args[0]).split(",")]
        if set(names) <= _MODEL_ARGUMENTS:
            continue
        dimensions.append((names, list(mark.args[1])))

    if not dimensions:
        return [("", {})]

    expanded: list[tuple[str, dict[str, Any]]] = []
    for combination in product(*(values for _, values in dimensions)):
        params: dict[str, Any] = {}
        suffix_parts: list[str] = []
        for (names, _), value in zip(dimensions, combination, strict=True):
            if len(names) == 1:
                params[names[0]] = value
                suffix_parts.append(f"{names[0]}={_param_id(value)}")
                continue
            for name, item in zip(names, value, strict=True):
                params[name] = item
                suffix_parts.append(f"{name}={_param_id(item)}")
        expanded.append(("[" + ",".join(suffix_parts) + "]", params))
    return expanded


def _case_id(scenario: str, function_name: str, suffix: str) -> str:
    name = function_name.removeprefix("test_")
    return f"{scenario}.{name}{suffix}"


def _run_function(function: Callable[..., Any], kwargs: dict[str, Any]) -> None:
    result = function(**kwargs)
    if inspect.isawaitable(result):
        asyncio.run(result)


def _param_id(value: Any) -> str:
    if isinstance(value, str):
        normalized = value.replace(" ", "_")
        if len(normalized) <= 32:
            return normalized
        digest = hashlib.sha256(value.encode()).hexdigest()[:8]
        return f"{normalized[:23]}-{digest}"
    return str(value)
