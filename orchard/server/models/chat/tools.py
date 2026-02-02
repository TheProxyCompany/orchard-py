from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatCompletionToolChoice(BaseModel):
    """Defines a tool for the chat completion request."""

    class FunctionName(BaseModel):
        """Defines a function name for the chat completion tool."""

        name: str = Field(description="The name of the function to call.")

    type: Literal["function"] = "function"
    function: FunctionName = Field(description="The function to call.")

    def to_dict(self):
        return {"type": "function", "name": self.function.name}


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
