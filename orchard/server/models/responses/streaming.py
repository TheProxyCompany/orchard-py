"""Open Responses streaming event models.

Maps PSE state events to Open Responses SSE events.
"""

from typing import Literal

from pydantic import BaseModel, Field

from orchard.server.models.responses.output import (
    IncompleteDetails,
    OutputFunctionCall,
    OutputMessage,
    OutputReasoning,
    OutputStatus,
    OutputTextContent,
    ReasoningContent,
    ResponseUsage,
    generate_message_id,
    generate_response_id,
    generate_tool_call_id,
    get_current_timestamp,
)

# --- Response Lifecycle Events ---


class ResponseCreatedEvent(BaseModel):
    """Emitted when response generation begins."""

    type: Literal["response.created"] = "response.created"
    sequence_number: int
    response: "ResponseSnapshot"


class ResponseInProgressEvent(BaseModel):
    """Emitted when response generation is actively in progress."""

    type: Literal["response.in_progress"] = "response.in_progress"
    sequence_number: int
    response: "ResponseSnapshot"


class ResponseCompletedEvent(BaseModel):
    """Emitted when response generation completes successfully."""

    type: Literal["response.completed"] = "response.completed"
    sequence_number: int
    response: "ResponseSnapshot"


class ResponseFailedEvent(BaseModel):
    """Emitted when response generation fails."""

    type: Literal["response.failed"] = "response.failed"
    sequence_number: int
    response: "ResponseSnapshot"


class ResponseIncompleteEvent(BaseModel):
    """Emitted when response generation stops before completion."""

    type: Literal["response.incomplete"] = "response.incomplete"
    sequence_number: int
    response: "ResponseSnapshot"


# --- Output Item Events ---


class OutputItemAddedEvent(BaseModel):
    """Emitted when a new output item starts."""

    type: Literal["response.output_item.added"] = "response.output_item.added"
    sequence_number: int
    output_index: int
    item: OutputMessage | OutputFunctionCall | OutputReasoning


class OutputItemDoneEvent(BaseModel):
    """Emitted when an output item is complete."""

    type: Literal["response.output_item.done"] = "response.output_item.done"
    sequence_number: int
    output_index: int
    item: OutputMessage | OutputFunctionCall | OutputReasoning


# --- Content Part Events ---


class ContentPartAddedEvent(BaseModel):
    """Emitted when a new content part starts within an item."""

    type: Literal["response.content_part.added"] = "response.content_part.added"
    sequence_number: int
    item_id: str
    output_index: int
    content_index: int
    part: OutputTextContent | ReasoningContent


class ContentPartDoneEvent(BaseModel):
    """Emitted when a content part is complete."""

    type: Literal["response.content_part.done"] = "response.content_part.done"
    sequence_number: int
    item_id: str
    output_index: int
    content_index: int
    part: OutputTextContent | ReasoningContent


# --- Text Delta Events ---


class OutputTextDeltaEvent(BaseModel):
    """Emitted for each text chunk in an assistant message."""

    type: Literal["response.output_text.delta"] = "response.output_text.delta"
    sequence_number: int
    item_id: str
    output_index: int
    content_index: int
    delta: str
    logprobs: list = Field(default_factory=list)


class OutputTextDoneEvent(BaseModel):
    """Emitted when text generation for a content part is complete."""

    type: Literal["response.output_text.done"] = "response.output_text.done"
    sequence_number: int
    item_id: str
    output_index: int
    content_index: int
    text: str
    logprobs: list = Field(default_factory=list)


# --- Function Call Argument Events ---


class FunctionCallArgumentsDeltaEvent(BaseModel):
    """Emitted for each chunk of function call arguments."""

    type: Literal["response.function_call_arguments.delta"] = (
        "response.function_call_arguments.delta"
    )
    sequence_number: int
    item_id: str
    output_index: int
    delta: str


class FunctionCallArgumentsDoneEvent(BaseModel):
    """Emitted when function call arguments are complete."""

    type: Literal["response.function_call_arguments.done"] = (
        "response.function_call_arguments.done"
    )
    sequence_number: int
    item_id: str
    output_index: int
    arguments: str


# --- Reasoning Events ---


class ReasoningDeltaEvent(BaseModel):
    """Emitted for each chunk of reasoning content."""

    type: Literal["response.reasoning.delta"] = "response.reasoning.delta"
    sequence_number: int
    item_id: str
    output_index: int
    content_index: int
    delta: str


class ReasoningDoneEvent(BaseModel):
    """Emitted when reasoning content is complete."""

    type: Literal["response.reasoning.done"] = "response.reasoning.done"
    sequence_number: int
    item_id: str
    output_index: int
    content_index: int
    text: str


class ReasoningSummaryTextDeltaEvent(BaseModel):
    """Emitted for each chunk of reasoning summary."""

    type: Literal["response.reasoning_summary_text.delta"] = (
        "response.reasoning_summary_text.delta"
    )
    sequence_number: int
    item_id: str
    output_index: int
    summary_index: int
    delta: str


class ReasoningSummaryTextDoneEvent(BaseModel):
    """Emitted when reasoning summary is complete."""

    type: Literal["response.reasoning_summary_text.done"] = (
        "response.reasoning_summary_text.done"
    )
    sequence_number: int
    item_id: str
    output_index: int
    summary_index: int
    text: str


# --- Error Event ---


class StreamErrorEvent(BaseModel):
    """Emitted when an error occurs during streaming."""

    type: Literal["error"] = "error"
    sequence_number: int
    error: dict


# --- Response Snapshot (partial state during streaming) ---


class ResponseSnapshot(BaseModel):
    """Snapshot of response state at a point in the stream."""

    id: str = Field(default_factory=generate_response_id)
    object: Literal["response"] = "response"
    created_at: int = Field(default_factory=get_current_timestamp)
    completed_at: int | None = None
    status: OutputStatus = OutputStatus.IN_PROGRESS
    incomplete_details: IncompleteDetails | None = None
    model: str
    output: list[OutputMessage | OutputFunctionCall | OutputReasoning] = Field(
        default_factory=list
    )
    usage: ResponseUsage | None = None


# --- Stream State Tracker ---


class StreamingOutputItem:
    """Tracks state for a single output item during streaming."""

    def __init__(
        self,
        item_id: str,
        output_index: int,
        item_type: str,
        call_id: str | None = None,
        function_name: str | None = None,
    ):
        self.item_id = item_id
        self.output_index = output_index
        self.item_type = item_type
        self.call_id = call_id
        self.function_name = function_name
        self.accumulated_content = ""
        self.content_index = 0
        self.status = OutputStatus.IN_PROGRESS

    def to_skeleton(self) -> OutputMessage | OutputFunctionCall | OutputReasoning:
        """Create the skeleton item for output_item.added event."""
        if self.item_type == "message":
            return OutputMessage(
                id=self.item_id,
                status=OutputStatus.IN_PROGRESS,
                content=[],
            )
        elif self.item_type == "function_call":
            return OutputFunctionCall(
                id=self.item_id,
                call_id=self.call_id or generate_tool_call_id(),
                name=self.function_name or "",
                arguments="",
                status=OutputStatus.IN_PROGRESS,
            )
        else:  # reasoning
            return OutputReasoning(
                id=self.item_id,
                status=OutputStatus.IN_PROGRESS,
                content=[],
                summary=[],
            )

    def to_completed(self) -> OutputMessage | OutputFunctionCall | OutputReasoning:
        """Create the completed item for output_item.done event."""
        if self.item_type == "message":
            return OutputMessage(
                id=self.item_id,
                status=OutputStatus.COMPLETED,
                content=[OutputTextContent(text=self.accumulated_content)],
            )
        elif self.item_type == "function_call":
            return OutputFunctionCall(
                id=self.item_id,
                call_id=self.call_id or generate_tool_call_id(),
                name=self.function_name or "",
                arguments=self.accumulated_content,
                status=OutputStatus.COMPLETED,
            )
        else:  # reasoning
            return OutputReasoning(
                id=self.item_id,
                status=OutputStatus.COMPLETED,
                content=[ReasoningContent(text=self.accumulated_content)],
                summary=[],
            )


class ResponseStreamState:
    """Tracks the complete state of a response during streaming."""

    def __init__(self, response_id: str, model: str):
        self.response_id = response_id
        self.model = model
        self.created_at = get_current_timestamp()
        self.completed_at: int | None = None
        self.items: dict[int, StreamingOutputItem] = {}  # output_index -> item
        self.sequence_number = 0
        self.status = OutputStatus.IN_PROGRESS
        self.incomplete_details: IncompleteDetails | None = None
        self.usage: ResponseUsage | None = None

    def next_sequence_number(self) -> int:
        """Get and increment the sequence number."""
        seq = self.sequence_number
        self.sequence_number += 1
        return seq

    def get_or_create_item(
        self,
        output_index: int,
        item_type: str,
        identifier: str | None = None,
    ) -> StreamingOutputItem:
        """Get existing item or create new one for the given output_index."""
        if output_index not in self.items:
            item_id = generate_message_id()
            call_id = None
            function_name = None

            if item_type == "function_call" and identifier:
                # identifier format: "tool_call:function_name"
                call_id = generate_tool_call_id()
                if identifier.startswith("tool_call:"):
                    function_name = identifier[len("tool_call:") :]

            self.items[output_index] = StreamingOutputItem(
                item_id=item_id,
                output_index=output_index,
                item_type=item_type,
                call_id=call_id,
                function_name=function_name,
            )
        return self.items[output_index]

    def snapshot(self) -> ResponseSnapshot:
        """Create a snapshot of current response state."""
        output_items = [
            item.to_completed()
            if item.status == OutputStatus.COMPLETED
            else item.to_skeleton()
            for item in sorted(self.items.values(), key=lambda x: x.output_index)
        ]
        return ResponseSnapshot(
            id=self.response_id,
            created_at=self.created_at,
            completed_at=self.completed_at,
            status=self.status,
            incomplete_details=self.incomplete_details,
            model=self.model,
            output=output_items,
            usage=self.usage,
        )


# Union of all streaming event types
StreamingEvent = (
    ResponseCreatedEvent
    | ResponseInProgressEvent
    | ResponseCompletedEvent
    | ResponseFailedEvent
    | ResponseIncompleteEvent
    | OutputItemAddedEvent
    | OutputItemDoneEvent
    | ContentPartAddedEvent
    | ContentPartDoneEvent
    | OutputTextDeltaEvent
    | OutputTextDoneEvent
    | FunctionCallArgumentsDeltaEvent
    | FunctionCallArgumentsDoneEvent
    | ReasoningDeltaEvent
    | ReasoningDoneEvent
    | ReasoningSummaryTextDeltaEvent
    | ReasoningSummaryTextDoneEvent
    | StreamErrorEvent
)
