"""Whole generated token counts conserve a hard cap even without visible text."""

from __future__ import annotations

import asyncio
import json

import pytest

from orchard.clients.client import Client
from orchard.clients.responses import (
    ResponsesRequest,
    aggregate_non_streaming_response,
    iter_response_events,
)
from orchard.engine import ClientDelta
from orchard.server.models.chat.output import ChatCompletionUsage
from orchard.server.routes import responses as response_route


class Ipc:
    async def next_delta(self, queue):
        return await queue.get()


def deltas(generated: int, reasoning: int):
    # A later metadata-only final delta must retain the earlier prompt count,
    # without subtracting or double-counting its reasoning subset.
    return [
        {
            "request_id": 1,
            "prompt_token_count": 17,
            "cached_token_count": 4,
            "generation_len": 1,
            "reasoning_tokens": 1,
            "is_final_delta": False,
        },
        {
            "request_id": 1,
            "generation_len": generated,
            "reasoning_tokens": reasoning,
            "is_final_delta": True,
            "finish_reason": "length",
        },
    ]


async def stream(items):
    for item in items:
        yield item


async def queue(items):
    result = asyncio.Queue()
    for item in items:
        await result.put(item)
    return result


@pytest.mark.parametrize(("generated", "reasoning"), [(5, 5), (512, 496)])
def test_sdk_chat_and_responses_count_whole_generation(generated, reasoning):
    items = deltas(generated, reasoning)
    chat = Client._extract_usage([ClientDelta(**item) for item in items])
    assert (
        chat.prompt_tokens,
        chat.completion_tokens,
        chat.reasoning_tokens,
        chat.total_tokens,
    ) == (17, generated, reasoning, 17 + generated)
    response = aggregate_non_streaming_response(
        items, "fixture", ResponsesRequest.from_text("hello")
    )
    assert response.usage is not None
    assert (
        response.usage.input_tokens,
        response.usage.output_tokens,
        response.usage.total_tokens,
    ) == (17, generated, 17 + generated)
    assert response.usage.output_tokens_details.reasoning_tokens == reasoning
    assert response.usage.input_tokens_details.cached_tokens == 4


@pytest.mark.asyncio
@pytest.mark.parametrize(("generated", "reasoning"), [(5, 5), (512, 496)])
async def test_responses_sdk_and_http_streams_and_http_aggregate_agree(
    generated, reasoning
):
    items = deltas(generated, reasoning)
    events = [
        event async for event in iter_response_events(stream(items), model_id="fixture")
    ]
    final = next(
        event.response for event in events if event.type == "response.incomplete"
    )
    assert (
        final.usage.input_tokens,
        final.usage.output_tokens,
        final.usage.total_tokens,
    ) == (17, generated, 17 + generated)
    assert final.usage.output_tokens_details.reasoning_tokens == reasoning

    result = await response_route.gather_non_streaming_response(
        1, await queue(items), Ipc()
    )
    assert (
        result["prompt_tokens"],
        result["completion_tokens"],
        result["reasoning_tokens"],
        result["total_tokens"],
    ) == (17, generated, reasoning, 17 + generated)
    server_events = [
        event
        async for event in response_route.stream_response_generator(
            1, await queue(items), Ipc(), "resp_fixture", "fixture"
        )
    ]
    final = next(
        json.loads(event["data"])["response"]
        for event in server_events
        if event.get("event") == "response.incomplete"
    )
    usage = final["usage"]
    assert (usage["input_tokens"], usage["output_tokens"], usage["total_tokens"]) == (
        17,
        generated,
        17 + generated,
    )
    assert usage["output_tokens_details"]["reasoning_tokens"] == reasoning
    assert usage["input_tokens_details"]["cached_tokens"] == 4


def test_http_chat_keeps_legacy_visible_field_and_canonical_generated_alias():
    usage = ChatCompletionUsage(
        input_tokens=17, output_tokens=0, reasoning_tokens=5, total_tokens=22
    )
    result = usage.model_dump()
    assert result["output_tokens"] == 0
    assert result["completion_tokens"] == 5
    assert result["completion_tokens_details"]["reasoning_tokens"] == 5
    assert (
        result["total_tokens"] == result["prompt_tokens"] + result["completion_tokens"]
    )
