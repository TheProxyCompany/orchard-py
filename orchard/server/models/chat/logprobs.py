from __future__ import annotations

from pydantic import BaseModel, Field, model_serializer

# --- Constants ---
LOGPROB_PRECISION = 6


class ChatCompletionLogProbs(BaseModel):
    """Represents the log probabilities of each token in the response."""

    class LogProbsContent(BaseModel):
        content: list[ChatCompletionLogProbs] | None = Field(
            default=None,
            description="A list of message content tokens with log probability information.",
        )

    bytes: list[int] | None = Field(
        default=None,
        description="A list of integers representing the UTF-8 bytes representation of the token.",
    )
    token: str = Field(description="The token.")
    logprob: float = Field(description="The log probability of this token.")
    top_logprobs: list[ChatCompletionLogProbs] | None = Field(
        default=None,
        description="The top log probabilities for this token.",
    )

    @model_serializer
    def serialize_model(self):
        result = {
            "token": self.token,
            "logprob": round(self.logprob, LOGPROB_PRECISION),
            "bytes": self.bytes,
        }
        if self.top_logprobs:
            result["top_logprobs"] = self.top_logprobs
        return result
