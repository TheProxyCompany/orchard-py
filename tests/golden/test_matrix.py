import pytest

from tests.golden.cases.registry import (
    format_failures,
    model_cases,
    pipeline_cases,
    run_cases,
)
from tests.models import MODELS, PIPELINE_TOOL_MODELS, Model

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.template_type)
async def test_golden_correctness_for_model(client, model: Model):
    failures = await run_cases(model_cases(model), {"client": client}, model)
    assert not failures, format_failures(model.template_type, failures)


async def test_golden_pipeline_cases(client, engine):
    # Hydrate the complete heterogeneous tool set concurrently before the
    # cases fan out. Model activation under unified-memory pressure is part of
    # the behavior this release-facing suite must exercise.
    await engine.load_models(PIPELINE_TOOL_MODELS)
    failures = await run_cases(pipeline_cases(), {"client": client})
    assert not failures, format_failures("pipeline", failures)
