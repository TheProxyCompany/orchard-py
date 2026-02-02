from typing import Literal

from pydantic import BaseModel, Field

from orchard.server.models.tools import ToolUseMode


class Function(BaseModel):
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

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": "object",
            "description": self.description or self.name,
            "properties": {
                "name": {"const": self.name},
                "arguments": self.parameters,
            },
            "strict": self.strict,
            "required": ["name", "arguments"],
        }


class FunctionID(BaseModel):
    """Defines a function tool for the response request."""

    type: Literal["function"] = "function"
    name: str = Field(description="The name of the function to call.")

    def to_dict(self):
        return self.model_dump()
