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
    failures = await run_cases(model_cases(), {"client": client}, model)
    assert not failures, format_failures(model.template_type, failures)


async def test_golden_pipeline_cases(client, engine):
    # Hydrate the tool models one at a time before the cases fan out:
    # letting the concurrent first wave activate several diffusion/TTS
    # models on demand — on top of the resident chat matrix — spikes wired
    # memory past the Metal limit and requests fail (or the engine dies
    # with a silent abort). Same pacing the buckshot preload uses.
    for model_id in PIPELINE_TOOL_MODELS:
        await engine.load_models([model_id])
    failures = await run_cases(pipeline_cases(), {"client": client})
    assert not failures, format_failures("pipeline", failures)
