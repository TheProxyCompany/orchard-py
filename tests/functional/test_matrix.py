import pytest

from tests.functional.cases.registry import (
    cases_for_model,
    format_failures,
    run_cases,
)
from tests.models import MODELS, Model

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.template_type)
async def test_functional_correctness_for_model(
    live_server,
    client,
    engine,
    model: Model,
):
    cases = cases_for_model(model)
    failures, skipped = await run_cases(
        cases,
        {
            "live_server": live_server,
            "client": client,
            "engine": engine,
        },
        model,
    )
    if skipped:
        print("\n".join(f"[skipped] {item}" for item in skipped), flush=True)
    assert not failures, format_failures(model, failures)
