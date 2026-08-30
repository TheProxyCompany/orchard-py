from tests.functional.cases.registry import _param_id
from tests.functional.cases.registry import cases_for_model as functional_cases
from tests.golden.cases.registry import model_cases, pipeline_cases
from tests.models import MODELS, PIPELINE_TOOL_MODELS


def test_buckshot_population_has_exact_unique_case_identities() -> None:
    case_ids = [
        *(
            f"functional::{model.template_type}::{case.id}"
            for model in MODELS
            for case in functional_cases(model)
        ),
        *(
            f"golden::{model.template_type}::{case.id}"
            for model in MODELS
            for case in model_cases(model)
        ),
        *(f"golden::pipeline::{case.id}" for case in pipeline_cases()),
    ]

    assert len(case_ids) == 579
    assert len(case_ids) == len(set(case_ids))


def test_buckshot_model_union_is_exact_and_duplicate_free() -> None:
    model_ids = [
        *dict.fromkeys([*(model.checkpoint for model in MODELS), *PIPELINE_TOOL_MODELS])
    ]

    assert len(model_ids) == 18
    assert len(model_ids) == len(set(model_ids))


def test_buckshot_request_plan_counts_actual_attempts_by_dependent_phase() -> None:
    plans = [
        *(case.request_phases for model in MODELS for case in functional_cases(model)),
        *(case.request_phases for model in MODELS for case in model_cases(model)),
        *(case.request_phases for case in pipeline_cases()),
    ]
    width = max(map(len, plans))
    phase_counts = [
        sum(phases[phase] if phase < len(phases) else 0 for phases in plans)
        for phase in range(width)
    ]

    assert phase_counts == [586, 97, 34, 2]
    assert sum(phase_counts) == 719


def test_long_parameter_ids_keep_a_collision_resistant_suffix() -> None:
    prefix = "You have 5 output tokens. Respond"
    first = _param_id(f"{prefix} with exactly five words")
    second = _param_id(f"{prefix} with a five-token plea")

    assert first != second
    assert len(first) == 32
    assert len(second) == 32
