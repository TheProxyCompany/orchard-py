"""Explicit budgets for tests that require a completed answer or tool call.

LFM2.5 requires reasoning even when a caller requests reasoning=False. The old
engine gave reasoning up to 8,192 tokens outside max_generated_tokens. The new
engine correctly counts them; semantic tests must now request that room
explicitly. Short-cap/truncation tests deliberately do not use this helper.
"""

from tests.models import MODELS

REQUIRED_REASONING_ALLOWANCE = 8192


def requires_reasoning(model_id: str) -> bool:
    return any(model.checkpoint == model_id and model.thinking == "required" for model in MODELS)


def semantic_token_limit(model_id: str, answer_tokens: int) -> int:
    if answer_tokens <= 0:
        raise ValueError("answer_tokens must be positive")
    return answer_tokens + (REQUIRED_REASONING_ALLOWANCE if requires_reasoning(model_id) else 0)
