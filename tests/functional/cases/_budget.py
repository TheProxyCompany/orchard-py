"""Generation budgets for semantic tests of mandatory-thinking models."""

from tests.models import MODELS

_REQUIRED_THINKING = {
    model.checkpoint for model in MODELS if model.thinking == "required"
}


def completion_budget(model_id: str, visible_budget: int) -> int:
    """Leave room for private reasoning before checking visible output.

    Tests that specifically check a small cap should use the literal cap instead.
    """
    return (
        max(512, visible_budget) if model_id in _REQUIRED_THINKING else visible_budget
    )
