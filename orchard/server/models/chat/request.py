from __future__ import annotations

from typing import Any

from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    field_validator,
    model_serializer,
    model_validator,
)

from orchard.server.models.chat.format import (
    ChatCompletionJsonObjectResponseFormat,
    ChatCompletionJSONSchemaResponseFormat,
    ChatCompletionTextResponseFormat,
)
from orchard.server.models.chat.tools import (
    ChatCompletionTool,
    ChatCompletionToolChoice,
)
from orchard.server.models.tools import ToolCall, ToolUseMode
from orchard.server.models.reasoning import normalize_reasoning_value

ReasoningInput = bool | str | dict[str, Any] | None


class ChatMessage(BaseModel):
    """Represents a single message within the chat conversation."""

    role: str | None = Field(default="", description="The role of the messages author.")
    content: str | None = Field(description="The contents of the message.")
    tool_calls: list[ToolCall] = Field(
        default_factory=list,
        description="The tool calls that were made in the message.",
    )

    @model_serializer
    def serialize_model(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.role:
            result["role"] = self.role
        if self.content is not None:
            result["content"] = self.content
        if self.tool_calls:
            result["tool_calls"] = [
                tool_call.model_dump() for tool_call in self.tool_calls
            ]

        return result


class ChatCompletionRequest(BaseModel):
    """Defines the request schema for the chat completion endpoint."""

    _batch_size: int = PrivateAttr(default=1)
    _normalized_fields: dict[str, list[Any]] = PrivateAttr(default_factory=dict)

    model: str = Field(
        description="The identifier of the model designated for completion generation."
    )
    messages: list[list[ChatMessage]] | list[ChatMessage] = Field(
        description="A list of message histories. Accepts a single history or a list of histories.",
        min_length=1,
    )
    max_completion_tokens: int | list[int] | None = Field(
        default=None,
        description=(
            "The upper limit on the number of tokens to generate per completion. "
            "Accepts a single value or a list matching the batch size."
        ),
    )
    temperature: float | list[float] = Field(
        default=1.0,
        description="Controls randomness via sampling temperature.",
    )
    top_p: float | list[float] | None = Field(
        default=1.0,
        description="Implements nucleus sampling.",
    )
    top_k: int | list[int] | None = Field(
        default=None,
        description="Controls the number of tokens considered at each step.",
    )
    min_p: float | list[float] | None = Field(
        default=0.0,
        description="Minimum probability threshold for token consideration.",
    )
    logprobs: bool | list[bool] | None = Field(
        default=False,
        description="Whether to include the log probabilities of each token in the response.",
    )
    parallel_tool_calls: bool | None = Field(
        default=None,
        description="Whether to allow the model to run tool calls in parallel.",
    )
    tool_choice: ToolUseMode | ChatCompletionToolChoice | None = Field(
        default=None,
        description="Controls which (if any) tool is called by the model.",
    )
    tools: list[ChatCompletionTool] | list[list[ChatCompletionTool]] | None = Field(
        default=None,
        description="A list of tools that the model can use to generate a response.",
    )
    top_logprobs: int | list[int] | None = Field(
        default=None,
        description="The number of top log probabilities to include in the response.",
    )
    response_format: (
        ChatCompletionTextResponseFormat
        | ChatCompletionJSONSchemaResponseFormat
        | ChatCompletionJsonObjectResponseFormat
        | None
    ) | list[
        ChatCompletionTextResponseFormat
        | ChatCompletionJSONSchemaResponseFormat
        | ChatCompletionJsonObjectResponseFormat
        | None
    ] = Field(
        default=None,
        description="The format of the response.",
    )
    stop: str | list[str] | list[list[str]] | None = Field(
        default=None,
        description="A list of tokens to stop generation of the response. The returned text will not contain the stop sequence.",
    )
    task: str | list[str] | None = Field(
        default=None,
        description="Optional specialized task identifier for the decoder.",
    )
    reasoning: ReasoningInput | list[ReasoningInput] = Field(
        default=None,
        description=(
            "Flexible reasoning control. Accepts booleans, effort strings, "
            "or objects like {'effort': 'medium'}."
        ),
    )
    reasoning_effort: str | list[str] | None = Field(
        default=None,
        description=(
            "Deprecated in favor of `reasoning`. Retained for backward compatibility."
        ),
    )

    stream: bool | None = Field(
        default=False,
        description="Whether to stream the response to the client using Server-Sent Events.",
    )
    stream_options: ChatCompletionStreamOptions | None = Field(
        default=None,
        description="Additional options for streaming the response.",
    )
    best_of: int | list[int] | None = Field(
        default=None,
        description=(
            "Generates `best_of` completions server-side and returns the best `n` "
            "candidates measured by average log probability per token. "
            "Must be greater than or equal to `n`."
        ),
    )
    n: int = Field(
        default=1,
        ge=1,
        description="Number of candidate completions to generate for each prompt.",
    )

    @model_validator(mode="before")
    @classmethod
    def _prevalidate_batched_parameters(cls, data: Any) -> Any:
        """Early, context-aware validation for vectorized parameters.

        - Determines batch size from the shape of `messages`.
        - For vectorized params, validates each element against required ranges.
        - Also validates singleton values against the same ranges.
        """
        if not isinstance(data, dict):
            return data

        def coerce_float(name: str, value: Any) -> float:
            if isinstance(value, int | float) and not isinstance(value, bool):
                return float(value)
            if isinstance(value, str):
                try:
                    return float(value.strip())
                except Exception as exc:
                    raise ValueError(
                        f"'{name}' must be a number, got {value!r}"
                    ) from exc
            raise ValueError(f"'{name}' must be a number, got {type(value).__name__}")

        def coerce_int(name: str, value: Any) -> int:
            if isinstance(value, bool):
                raise ValueError(f"'{name}' must be an integer, got bool")
            if isinstance(value, int):
                return value
            if isinstance(value, str):
                s = value.strip()
                if s.isdigit() or (s.startswith("+") and s[1:].isdigit()):
                    return int(s)
                raise ValueError(f"'{name}' must be an integer, got {value!r}")
            raise ValueError(f"'{name}' must be an integer, got {type(value).__name__}")

        def validate_numeric(
            name: str,
            raw: Any,
            *,
            minimum: float | int | None,
            maximum: float | int | None,
            integer: bool = False,
            optional: bool = True,
        ) -> None:
            if raw is None:
                if optional:
                    return
                raise ValueError(f"'{name}' is required")

            def _check_once(v: Any) -> None:
                if integer:
                    val = coerce_int(name, v)
                else:
                    val = coerce_float(name, v)
                if minimum is not None and val < minimum:
                    raise ValueError(f"'{name}' out of range: {val} < {minimum}")
                if maximum is not None and val > maximum:
                    raise ValueError(f"'{name}' out of range: {val} > {maximum}")

            if isinstance(raw, list):
                for idx, item in enumerate(raw):
                    if item is None:
                        if optional:
                            continue
                        raise ValueError(f"'{name}[{idx}]' is required")
                    _check_once(item)
            else:
                _check_once(raw)

        # Validate ranges for vectorized params
        validate_numeric(
            "temperature",
            data.get("temperature"),
            minimum=0.0,
            maximum=2.0,
            integer=False,
            optional=False,
        )
        validate_numeric(
            "top_p",
            data.get("top_p"),
            minimum=0.0,
            maximum=1.0,
            integer=False,
            optional=True,
        )
        validate_numeric(
            "min_p",
            data.get("min_p"),
            minimum=0.0,
            maximum=1.0,
            integer=False,
            optional=True,
        )
        validate_numeric(
            "top_k",
            data.get("top_k"),
            minimum=1,
            maximum=100,
            integer=True,
            optional=True,
        )
        validate_numeric(
            "top_logprobs",
            data.get("top_logprobs"),
            minimum=0,
            maximum=20,
            integer=True,
            optional=True,
        )
        validate_numeric(
            "best_of",
            data.get("best_of"),
            minimum=1,
            maximum=None,
            integer=True,
            optional=True,
        )
        # Validate singleton numeric even if not vectorized
        if "max_completion_tokens" in data:
            validate_numeric(
                "max_completion_tokens",
                data.get("max_completion_tokens"),
                minimum=1,
                maximum=None,
                integer=True,
                optional=True,
            )

        return data

    @model_validator(mode="before")
    @classmethod
    def _migrate_reasoning_inputs(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        reasoning = data.get("reasoning")
        reasoning_effort = data.get("reasoning_effort")
        if reasoning in (None, [], {}):
            if reasoning_effort not in (None, [], {}):
                data["reasoning"] = reasoning_effort
        return data

    @field_validator("messages", mode="after")
    @classmethod
    def _normalize_messages(
        cls, value: list[list[ChatMessage]] | list[ChatMessage]
    ) -> list[list[ChatMessage]]:
        if not value:
            raise ValueError("messages must contain at least one conversation history")
        first = value[0]
        if isinstance(first, ChatMessage):
            assert isinstance(value, list)
            return [list(value)]  # type: ignore[return-value]
        normalized: list[list[ChatMessage]] = []
        for conversation in value:  # type: ignore[assignment]
            if not conversation:
                raise ValueError(
                    "each conversation history must contain at least one message"
                )
            normalized.append(list(conversation))  # type: ignore[arg-type]
        return normalized

    @staticmethod
    def _broadcast_list(
        value: Any,
        batch_size: int,
        field_name: str,
    ) -> list[Any]:
        if value is None:
            return [None] * batch_size
        if isinstance(value, list):
            if len(value) == batch_size:
                return list(value)
            if len(value) == 1:
                return [value[0]] * batch_size
            raise ValueError(
                f"Length of '{field_name}' ({len(value)}) does not match batch size {batch_size}."
            )
        return [value] * batch_size

    @staticmethod
    def _normalize_stop_sequences(value: Any, batch_size: int) -> list[list[str]]:
        if value is None:
            return [[] for _ in range(batch_size)]
        if isinstance(value, str):
            return [[value] for _ in range(batch_size)]
        if isinstance(value, list):
            if not value:
                return [[] for _ in range(batch_size)]
            # list of strings → broadcast
            if all(isinstance(item, str) for item in value):
                base = [str(item) for item in value]
                return [list(base) for _ in range(batch_size)]
            # list of per-instance specifications
            if len(value) == 1:
                single = value[0]
                if single is None:
                    return [[] for _ in range(batch_size)]
                if isinstance(single, str):
                    return [[single] for _ in range(batch_size)]
                if isinstance(single, list) and all(
                    isinstance(item, str) for item in single
                ):
                    base = [str(item) for item in single]
                    return [list(base) for _ in range(batch_size)]
                raise ValueError("Invalid stop sequence specification")
            if len(value) != batch_size:
                raise ValueError(
                    f"Length of 'stop' ({len(value)}) does not match batch size {batch_size}."
                )
            normalized: list[list[str]] = []
            for entry in value:
                if entry is None:
                    normalized.append([])
                elif isinstance(entry, str):
                    normalized.append([entry])
                elif isinstance(entry, list) and all(
                    isinstance(item, str) for item in entry
                ):
                    normalized.append([str(item) for item in entry])
                else:
                    raise ValueError("Invalid stop sequence specification")
            return normalized
        raise ValueError("Invalid stop sequence specification")

    @staticmethod
    def _normalize_tools(
        value: list[ChatCompletionTool] | list[list[ChatCompletionTool]] | None,
        batch_size: int,
    ) -> list[list[ChatCompletionTool] | None]:
        if value is None:
            return [None] * batch_size
        if (
            isinstance(value, list)
            and value
            and isinstance(value[0], ChatCompletionTool)
        ):
            return [list(value)] * batch_size  # type: ignore[return-value]
        if isinstance(value, list):
            if len(value) == 1:
                single = value[0]
                if single is None:
                    return [None] * batch_size
                if isinstance(single, list):
                    return [list(single)] * batch_size
            if len(value) != batch_size:
                raise ValueError(
                    f"Length of 'tools' ({len(value)}) does not match batch size {batch_size}."
                )
            normalized: list[list[ChatCompletionTool] | None] = []
            for entry in value:
                if entry is None:
                    normalized.append(None)
                else:
                    normalized.append(list(entry))  # type: ignore[arg-type]
            return normalized
        raise ValueError("Invalid tools specification")

    @staticmethod
    def _normalize_response_format(
        value: Any,
        batch_size: int,
    ) -> list[
        ChatCompletionTextResponseFormat
        | ChatCompletionJSONSchemaResponseFormat
        | ChatCompletionJsonObjectResponseFormat
        | None
    ]:
        if value is None:
            return [None] * batch_size
        if isinstance(value, list):
            if not value:
                return [None] * batch_size
            if len(value) == batch_size:
                return list(value)
            if len(value) == 1:
                return [value[0]] * batch_size
            raise ValueError(
                f"Length of 'response_format' ({len(value)}) does not match batch size {batch_size}."
            )
        return [value] * batch_size

    @model_validator(mode="after")
    def _broadcast_parameters(self) -> ChatCompletionRequest:
        batch_size = len(self.messages)
        object.__setattr__(self, "_batch_size", batch_size)

        normalized_fields: dict[str, list[Any]] = {}

        normalized_fields["max_completion_tokens"] = self._broadcast_list(
            self.max_completion_tokens, batch_size, "max_completion_tokens"
        )
        normalized_fields["temperature"] = self._broadcast_list(
            self.temperature, batch_size, "temperature"
        )
        normalized_fields["top_p"] = self._broadcast_list(
            self.top_p, batch_size, "top_p"
        )
        normalized_fields["top_k"] = self._broadcast_list(
            self.top_k, batch_size, "top_k"
        )
        normalized_fields["min_p"] = self._broadcast_list(
            self.min_p, batch_size, "min_p"
        )
        normalized_fields["logprobs"] = self._broadcast_list(
            self.logprobs, batch_size, "logprobs"
        )
        normalized_fields["top_logprobs"] = self._broadcast_list(
            self.top_logprobs, batch_size, "top_logprobs"
        )
        normalized_fields["tools"] = self._normalize_tools(self.tools, batch_size)
        normalized_fields["response_format"] = self._normalize_response_format(
            self.response_format, batch_size
        )
        normalized_fields["stop"] = self._normalize_stop_sequences(
            self.stop, batch_size
        )
        normalized_fields["task"] = self._broadcast_list(self.task, batch_size, "task")
        reasoning_inputs = self._broadcast_list(self.reasoning, batch_size, "reasoning")
        normalized_fields["reasoning_effort"] = [
            normalize_reasoning_value(value, field_name="reasoning")
            for value in reasoning_inputs
        ]
        normalized_fields["n"] = self._broadcast_list(self.n, batch_size, "n")

        raw_best_of = self._broadcast_list(self.best_of, batch_size, "best_of")
        best_of_values: list[int] = []
        stream_requested = bool(self.stream) if self.stream is not None else False
        for idx in range(batch_size):
            n_value = normalized_fields["n"][idx]
            candidate_best_of = raw_best_of[idx]
            if candidate_best_of is None:
                candidate_best_of = n_value
            if candidate_best_of < n_value:
                raise ValueError(
                    f"'best_of' ({candidate_best_of}) must be greater than or equal to 'n' ({n_value}) for batch index {idx}."
                )
            if stream_requested and candidate_best_of > n_value:
                raise ValueError(
                    "Parameter 'best_of' greater than 'n' is not supported when 'stream' is true."
                )
            best_of_values.append(int(candidate_best_of))
        normalized_fields["best_of"] = best_of_values

        object.__setattr__(self, "_normalized_fields", normalized_fields)
        return self

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def get_normalized_field(self, key: str) -> list[Any]:
        return self._normalized_fields[key]


class ChatCompletionStreamOptions(BaseModel):
    """Additional options for streaming the response."""

    include_usage: bool | None = Field(
        default=False,
        description="Whether to include the usage statistics in the response.",
    )
