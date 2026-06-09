import pytest

from tests.golden.cases.registry import (
    format_failures,
    model_cases,
    pipeline_cases,
    run_cases,
)
from tests.models import MODELS, Model

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.template_type)
async def test_golden_correctness_for_model(client, model: Model):
    failures = await run_cases(model_cases(), {"client": client}, model)
    assert not failures, format_failures(model.template_type, failures)


async def test_golden_pipeline_cases(client):
    failures = await run_cases(pipeline_cases(), {"client": client})
    assert not failures, format_failures("pipeline", failures)
