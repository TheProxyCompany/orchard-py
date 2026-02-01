import json
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from orchard.server.models.reasoning import (
    DEFAULT_BOOLEAN_REASONING_EFFORT,
    ReasoningEffort,
    normalize_reasoning_effort,
)
from orchard.server.models.responses.format import ResponseFormat
from orchard.server.models.responses.tools import (
    Function,
    FunctionID,
    ToolUseMode,
)

# --- Content Parts (for message content) ---


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
    detail: Literal["low", "high", "auto"] = Field(
        default="auto",
        description="Image detail level for processing.",
    )


ContentPart = Annotated[InputText | InputImage, Field(discriminator="type")]


# --- Input Item Types (discriminated by `type` field) ---


class InputMessageItem(BaseModel):
    """A message item in the input."""

    type: Literal["message"] = "message"
    role: Literal["user", "assistant", "system", "developer", "tool"] = Field(
        description="Role of the message author."
    )
    content: str | list[ContentPart] = Field(
        description="Message content as raw text or structured content parts."
    )


class InputFunctionCall(BaseModel):
    """A function call from a previous assistant turn."""

    type: Literal["function_call"] = "function_call"
    call_id: str = Field(description="Unique identifier for this function call.")
    name: str = Field(description="Name of the function that was called.")
    arguments: str = Field(description="JSON-encoded arguments passed to the function.")


class InputFunctionCallOutput(BaseModel):
    """The result of a function call."""

    type: Literal["function_call_output"] = "function_call_output"
    call_id: str = Field(
        description="The call_id of the function call this is responding to."
    )
    output: str = Field(description="The output of the function call.")


class InputReasoning(BaseModel):
    """Reasoning content from a previous turn."""

    type: Literal["reasoning"] = "reasoning"
    summary: list[dict] | None = Field(
        default=None,
        description="Summary of the reasoning (if available).",
    )
    encrypted_content: str | None = Field(
        default=None,
        description="Encrypted reasoning content for continuation.",
    )


InputItem = Annotated[
    InputMessageItem | InputFunctionCall | InputFunctionCallOutput | InputReasoning,
    Field(discriminator="type"),
]


# Legacy alias for backwards compatibility
class InputMessage(BaseModel):
    """Represents a single input message (legacy format)."""

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
    """Defines the request schema for the /v1/responses endpoint."""

    model: str = Field(description="Model ID used to generate the response.")
    input: str | list[InputItem] = Field(
        description="Input as a string (shorthand for user message) or array of items."
    )
    stream: bool | None = Field(
        default=None,
        description="Whether to stream the response.",
    )
    truncation: Literal["auto", "disabled"] | None = Field(
        default=None,
        description="Controls whether server can truncate input when it exceeds context window.",
    )
    parallel_tool_calls: bool | None = Field(
        default=None,
        description="Whether to allow the model to run tool calls in parallel.",
    )
    max_tool_calls: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of tool calls the model can emit in one response.",
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
    presence_penalty: float | None = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        description="Penalizes new tokens based on whether they appear in the text so far.",
    )
    frequency_penalty: float | None = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        description="Penalizes new tokens based on their frequency in the text so far.",
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
    top_logprobs: int | None = Field(
        default=None,
        ge=0,
        le=20,
        description="Number of top log probabilities to return per token.",
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
    include: list[str] | None = Field(
        default=None,
        description="What optional data to include in response (e.g., 'reasoning.encrypted_content', 'message.output_text.logprobs').",
    )
    metadata: dict[str, str] | None = Field(
        default=None,
        description="Key-value pairs attached to the response for tracking.",
    )
    # Persistence parameters - accepted but ignored (orchard-py is stateless)
    previous_response_id: str | None = Field(
        default=None,
        description="Ignored. Response chaining requires server-side storage.",
    )
    store: bool | None = Field(
        default=None,
        description="Ignored. Response persistence requires storage.",
    )
    background: bool | None = Field(
        default=None,
        description="Ignored. Async polling requires storage and task queue.",
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

    @model_validator(mode="after")
    def _normalize_string_input(self) -> "ResponseRequest":
        """Convert string input to a single user message item."""
        if isinstance(self.input, str):
            object.__setattr__(
                self,
                "input",
                [InputMessageItem(role="user", content=self.input)],
            )
        return self

    def get_message_items(self) -> list[InputMessageItem]:
        """Convert all input items to message items for template rendering.

        InputFunctionCall → assistant message with tool call JSON.
        InputFunctionCallOutput → tool message with output.
        InputReasoning → skipped (not representable as text).
        """
        if isinstance(self.input, str):
            return [InputMessageItem(role="user", content=self.input)]
        messages: list[InputMessageItem] = []
        for item in self.input:
            if isinstance(item, InputMessageItem):
                messages.append(item)
            elif isinstance(item, InputFunctionCall):
                call_json = json.dumps(
                    {"name": item.name, "arguments": item.arguments}
                )
                messages.append(
                    InputMessageItem(role="assistant", content=call_json)
                )
            elif isinstance(item, InputFunctionCallOutput):
                messages.append(
                    InputMessageItem(role="tool", content=item.output)
                )
        return messages
