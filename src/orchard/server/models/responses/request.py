from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

from proxy_inference_engine.server.models.reasoning import (
    DEFAULT_BOOLEAN_REASONING_EFFORT,
    ReasoningEffort,
    normalize_reasoning_effort,
)
from proxy_inference_engine.server.models.responses.format import ResponseFormat
from proxy_inference_engine.server.models.responses.tools import (
    Function,
    FunctionID,
    ToolUseMode,
)


class InputText(BaseModel):
    """Text content for an input message."""

    type: Literal["input_text"]
    text: str = Field(description="Raw text content.")


class InputImage(BaseModel):
    """Image content for an input message."""

    type: Literal["input_image"]
    image_url: str = Field(
        description="Data URL containing Base64-encoded image bytes."
    )


ContentPart = Annotated[InputText | InputImage, Field(discriminator="type")]


class InputMessage(BaseModel):
    """Represents a single input message."""

    role: str = Field(description="Role of the message author.")
    content: str | list[ContentPart] = Field(
        description="Message content as raw text or structured content parts."
    )


class ResponseReasoning(BaseModel):
    effort: ReasoningEffort = Field(
        description="Controls the depth of the reasoning phase.",
    )

    @classmethod
    def validate_effort(cls, value: ReasoningEffort) -> ReasoningEffort:
        normalized = normalize_reasoning_effort(value)
        assert normalized is not None
        return normalized

    @property
    def normalized_effort(self) -> ReasoningEffort:
        return self.validate_effort(self.effort)


class ResponseRequest(BaseModel):
    """Defines the request schema for the /v1/responses endpoint (MVP)."""

    model: str = Field(description="Model ID used to generate the response.")
    items: list[InputMessage] = Field(
        description="Ordered input messages for the request."
    )
    stream: bool | None = Field(
        default=None,
        description="Whether to stream the response.",
    )
    parallel_tool_calls: bool | None = Field(
        default=None,
        description="Whether to allow the model to run tool calls in parallel.",
    )
    instructions: str | None = Field(
        default=None,
        description="System/developer instructions for the model.",
    )
    max_output_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Upper bound for the number of tokens generated.",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature.",
    )
    top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling threshold.",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Controls the number of tokens considered at each step.",
    )
    min_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum probability threshold for token consideration.",
    )
    tool_choice: ToolUseMode | FunctionID = Field(
        default=ToolUseMode.AUTO,
        description="How the model should select which tool (or tools) to use when generating a response.",
    )
    tools: list[Function] | None = Field(
        default=None,
        description="A list of tools that the model can use to generate a response.",
    )
    text: ResponseFormat | None = Field(
        default=None,
        description="The format of the response.",
    )
    task: str | None = Field(
        default=None,
        description="Optional specialized task identifier for the decoder.",
    )
    reasoning: ResponseReasoning | bool | None = Field(
        default=None,
        description="Optional configuration for reasoning effort.",
    )

    @field_validator("reasoning", mode="before")
    @classmethod
    def _normalize_boolean_reasoning(cls, value: ResponseReasoning | bool | None):
        if value is None:
            return None
        if value is False:
            return None
        if value is True:
            return {"effort": DEFAULT_BOOLEAN_REASONING_EFFORT}
        return value
