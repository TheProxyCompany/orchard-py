from __future__ import annotations

import json
import secrets
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

TOOL_CALL_ID_PREFIX = "call-"


def generate_tool_call_id(prefix: str = TOOL_CALL_ID_PREFIX) -> str:
    """Generates a unique identifier string for a tool call."""
    random_part = secrets.token_urlsafe(22)
    return f"{prefix}{random_part}"


class ChatCompletionToolUsage(BaseModel):
    """Represents the usage of a tool in a chat completion."""

    class UsedFunction(BaseModel):
        """Represents a function that was used in a chat completion."""

        name: str | None = Field(
            default=None, description="The name of the function to call."
        )
        arguments: str = Field(
            default="",
            description="The arguments to pass to the function. JSON encoded.",
        )

    type: Literal["function"] = "function"
    id: str = Field(description="The unique identifier of the tool.")
    function: UsedFunction = Field(description="The function that was used.")


class ChatCompletionToolChoice(BaseModel):
    """Defines a tool for the chat completion request."""

    class FunctionName(BaseModel):
        """Defines a function name for the chat completion tool."""

        name: str = Field(description="The name of the function to call.")

    type: Literal["function"] = "function"
    function: FunctionName = Field(description="The function to call.")

    def to_dict(self):
        return {"type": "function", "name": self.function.name}


class ChatCompletionToolUseMode(Enum):
    """Controls which (if any) tool is called by the model."""

    AUTO = "auto"
    REQUIRED = "required"
    NONE = "none"

    def to_dict(self):
        return self.value


class ChatCompletionFunction(BaseModel):
    """Defines a function for the response request."""

    name: str = Field(description="The name of the function to call.")
    type: Literal["function"] = "function"
    description: str = Field(
        description="A description of the function. Used by the model to determine whether or not to call the function."
    )
    strict: bool = Field(
        default=True,
        description="Whether to enforce strict parameter validation.",
    )
    parameters: dict = Field(
        description="A JSON schema object describing the parameters of the function."
    )


class ChatCompletionTool(BaseModel):
    """Defines a tool for the chat completion request."""

    type: Literal["function"] = "function"
    function: ChatCompletionFunction = Field(description="The function to call.")

    def to_dict(self) -> dict:
        return {
            "name": self.function.name,
            "type": "object",
            "description": self.function.description or self.function.name,
            "properties": {
                "name": {"const": self.function.name},
                "arguments": self.function.parameters,
            },
            "strict": self.function.strict,
            "required": ["name", "arguments"],
        }
