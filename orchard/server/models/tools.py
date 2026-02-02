"""Shared tool-related types used across chat completions and responses endpoints."""

import secrets
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

TOOL_CALL_ID_PREFIX = "call_"


def generate_tool_call_id(prefix: str = TOOL_CALL_ID_PREFIX) -> str:
    """Generates a unique identifier string for a tool call."""
    random_part = secrets.token_urlsafe(22)
    return f"{prefix}{random_part}"


class ToolCallFunction(BaseModel):
    """The function invocation within a tool call."""

    name: str | None = Field(
        default=None, description="The name of the function to call."
    )
    arguments: str = Field(
        default="",
        description="The arguments to pass to the function. JSON encoded.",
    )


class ToolCall(BaseModel):
    """Represents a tool call made by the model."""

    type: Literal["function"] = "function"
    id: str = Field(
        default_factory=generate_tool_call_id,
        description="The unique identifier of the tool call.",
    )
    function: ToolCallFunction = Field(description="The function that was called.")


class ToolUseMode(Enum):
    """Controls which (if any) tool is called by the model."""

    AUTO = "auto"
    REQUIRED = "required"
    NONE = "none"

    def to_dict(self):
        return self.value
