import pytest

from tests.functional.cases._token_budget import (
    requires_reasoning,
    semantic_token_limit,
)
from tests.models import MODELS


def test_only_mandatory_reasoning_models_get_explicit_semantic_room():
    for model in MODELS:
        expected = 8192 + 64 if model.thinking == "required" else 64
        assert semantic_token_limit(model.checkpoint, 64) == expected
        assert requires_reasoning(model.checkpoint) == (model.thinking == "required")


def test_unknown_model_and_short_cap_remain_bounded():
    assert semantic_token_limit("unlisted/checkpoint", 5) == 5
    # This helper is opt-in; it cannot alter a caller's explicit cap.
    assert semantic_token_limit("LiquidAI/LFM2.5-8B-A1B", 128) == 8320
    with pytest.raises(ValueError):
        semantic_token_limit("LiquidAI/LFM2.5-8B-A1B", 0)


@pytest.mark.asyncio
async def test_short_total_cap_keeps_reasoning_and_answer_tokens_bounded():
    from orchard.engine import ClientResponse, UsageStats
    from tests.functional.cases.client import (
        test_client_chat_non_streaming as check_short_cap,
    )

    class RecordedClient:
        def __init__(self, visible, reasoning, text):
            self.response = ClientResponse(
                text=text,
                usage=UsageStats(
                    completion_tokens=visible + reasoning, reasoning_tokens=reasoning
                ),
            )

        async def achat(self, model, messages, **options):
            assert options["max_generated_tokens"] == 5
            return self.response

    required = "LiquidAI/LFM2.5-8B-A1B"
    optional = "google/gemma-4-E2B-it"
    await check_short_cap(RecordedClient(0, 5, ""), required, "test")
    await check_short_cap(RecordedClient(5, 0, "hello"), optional, "test")
    # The special handling must not admit an exceeded cap or hide an empty
    # answer from a model that can turn reasoning off.
    with pytest.raises(AssertionError):
        await check_short_cap(RecordedClient(0, 6, ""), required, "test")
    with pytest.raises(AssertionError):
        await check_short_cap(RecordedClient(0, 5, ""), optional, "test")
