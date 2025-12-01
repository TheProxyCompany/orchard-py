from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class UsageStats(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ClientDelta(BaseModel):
    request_id: int
    sequence_id: int | None = None
    prompt_index: int | None = None
    candidate_index: int | None = None
    prompt_token_count: int | None = None
    num_tokens_in_delta: int | None = None
    tokens: list[int] = Field(default_factory=list)
    top_logprobs: list[dict[str, float]] = Field(default_factory=list)
    cumulative_logprob: float | None = None
    generation_len: int | None = None
    content: str | None = None
    content_len: int | None = None
    inline_content_bytes: int | None = None
    modal_decoder_id: str | None = None
    modal_bytes_b64: str | None = None
    is_final: bool = Field(default=False, alias="is_final_delta")
    finish_reason: str | None = None

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)


class ClientResponse(BaseModel):
    text: str
    finish_reason: str | None = None
    usage: UsageStats = Field(default_factory=UsageStats)
    deltas: list[ClientDelta] = Field(default_factory=list)


__all__ = ["ClientDelta", "ClientResponse", "UsageStats"]
