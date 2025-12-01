from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

ReasoningEffort = Literal["minimal", "low", "medium", "high"]

_VALID_EFFORT_VALUES = {"minimal", "low", "medium", "high"}
DEFAULT_BOOLEAN_REASONING_EFFORT: ReasoningEffort = "medium"


def normalize_reasoning_effort(value: str | None) -> ReasoningEffort | None:
    """Lowercase and validate the reasoning effort string."""
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized not in _VALID_EFFORT_VALUES:
        raise ValueError(
            f"Invalid reasoning effort '{value}'. "
            f"Expected one of {sorted(_VALID_EFFORT_VALUES)}."
        )
    return normalized  # type: ignore[return-value]


def normalize_reasoning_value(
    value: Any,
    *,
    field_name: str = "reasoning",
) -> ReasoningEffort | None:
    """Normalize flexible reasoning inputs into a ReasoningEffort value."""
    if value is None:
        return None
    if isinstance(value, bool):
        return DEFAULT_BOOLEAN_REASONING_EFFORT if value else None
    if isinstance(value, str):
        return normalize_reasoning_effort(value)
    if isinstance(value, Mapping):
        if "effort" not in value:
            raise ValueError(f"'{field_name}' object must include an 'effort' field.")
        return normalize_reasoning_effort(value["effort"])
    raise ValueError(
        f"'{field_name}' must be a boolean, an effort string, "
        "or an object with an 'effort' field."
    )


__all__ = [
    "DEFAULT_BOOLEAN_REASONING_EFFORT",
    "ReasoningEffort",
    "normalize_reasoning_effort",
    "normalize_reasoning_value",
]
