"""Message text is assembled once, from one view of the reply.

Every token delta carries two views of the text. ``content`` is the decoded
text of the sampled tokens. The state events are the engine's text stream:
it holds back the end of the text while that end could still become a stop
sequence (or the newline in front of ``</think>``), never contains a stop
sequence, and hands held text over in a later delta. When a reply ends without
a stop (token limit, cancel) the held text arrives in the final delta, which
has no tokens and no ``content``.

A candidate that carries state events is assembled from its message
``content_delta`` spans alone. Mixing in ``content`` repeats text the spans
deliver later ("Hello the EE") and shows text the engine held back on purpose
(the stop sequence). A candidate that never carries state events has only
``content``.

The delta sequences below are written out in the shape the engine sends them
(one delta per sampled token, then a completion delta without tokens); no
engine runs here.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import pytest

from orchard.clients.client import Client
from orchard.clients.responses import (
    ResponsesRequest,
    aggregate_non_streaming_response,
    iter_response_events,
)
from orchard.engine import ClientDelta
from orchard.server.routes import chat as chat_route
from orchard.server.routes import responses as responses_route


def _event(event_type: str, **fields: Any) -> dict[str, Any]:
    return {
        "event_type": event_type,
        "item_type": "message",
        "output_index": 0,
        "identifier": "message",
        **fields,
    }


def _token(content: str, *events: dict[str, Any]) -> dict[str, Any]:
    return {
        "request_id": 1,
        "prompt_index": 0,
        "candidate_index": 0,
        "sequence_id": 7,
        "tokens": [11],
        "content": content,
        "state_events": list(events),
        "is_final_delta": False,
    }


def _final(finish_reason: str, *events: dict[str, Any]) -> dict[str, Any]:
    """The completion delta: no tokens, no content, only what the text stream releases."""
    return {
        "request_id": 1,
        "prompt_index": 0,
        "candidate_index": 0,
        "sequence_id": 7,
        "tokens": [],
        "content": "",
        "state_events": list(events),
        "is_final_delta": True,
        "finish_reason": finish_reason,
        "generation_len": 3,
    }


_HELLO = _token("Hello", _event("item_started"), _event("content_delta", delta="Hello"))


@dataclass(frozen=True)
class Reply:
    name: str
    deltas: list[dict[str, Any]]
    text: str


# Stop sequence "END", cut off by the token limit after " the E". "E" could
# start the stop sequence, so its span waits; the final delta releases it.
HELD_STOP_START = Reply(
    "length_limit_releases_held_stop_start",
    [
        _HELLO,
        _token(" the E", _event("content_delta", delta=" the ")),
        _final("length", _event("content_delta", delta="E")),
    ],
    "Hello the E",
)

# A bare newline is held in front of a possible "</think>": the token delta
# has no events at all, and the final delta releases the newline.
HELD_NEWLINE = Reply(
    "length_limit_releases_held_newline",
    [
        _HELLO,
        _token("\n"),
        _final("length", _event("content_delta", delta="\n")),
    ],
    "Hello\n",
)

# The reply ends on the stop sequence. Its tokens are decoded into content,
# but the text stream never shows them.
STOP_SEQUENCE = Reply(
    "stop_sequence_stays_out",
    [
        _HELLO,
        _token("EN"),
        _token("D", _event("item_completed", value="Hello")),
        _final("stop"),
    ],
    "Hello",
)

# Held text released in the middle of a reply: "e" of " the" could start
# "END", and the next token's span hands it over in front of its own text.
HELD_MID_REPLY = Reply(
    "held_text_released_mid_reply",
    [
        _HELLO,
        _token(" the", _event("content_delta", delta=" th")),
        _token(" cat", _event("content_delta", delta="e cat")),
        _final("stop"),
    ],
    "Hello the cat",
)

# An engine (or a model without a text state machine) that sends no state
# events: content is all there is.
NO_STATE_EVENTS = Reply(
    "no_state_events_uses_content",
    [
        {**_token("Hello"), "state_events": []},
        _token(" world"),
        _final("stop"),
    ],
    "Hello world",
)

WITH_STATE_EVENTS = [HELD_STOP_START, HELD_NEWLINE, STOP_SEQUENCE, HELD_MID_REPLY]
ALL_REPLIES = [*WITH_STATE_EVENTS, NO_STATE_EVENTS]


def _ids(replies: list[Reply]) -> list[str]:
    return [reply.name for reply in replies]


class _IpcStub:
    async def next_delta(self, queue: asyncio.Queue) -> dict:
        return await queue.get()


async def _queue_of(reply: Reply) -> asyncio.Queue:
    queue: asyncio.Queue = asyncio.Queue()
    for delta in reply.deltas:
        await queue.put(dict(delta))
    return queue


def _client_deltas(reply: Reply) -> list[ClientDelta]:
    return [ClientDelta(**delta) for delta in reply.deltas]


async def _iterate(reply: Reply):
    for delta in reply.deltas:
        yield delta


def _message_text(output: list[Any]) -> str:
    return "".join(
        part.text for item in output if item.type == "message" for part in item.content
    )


# --- /v1/chat/completions ---------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", ALL_REPLIES, ids=_ids(ALL_REPLIES))
async def test_chat_route_non_streaming(reply: Reply) -> None:
    result = await asyncio.wait_for(
        chat_route.gather_non_streaming_batch_response(
            1, await _queue_of(reply), _IpcStub(), [1], [1]
        ),
        timeout=1.0,
    )

    assert result["choices"][0].message.content == reply.text


@pytest.mark.asyncio
async def test_chat_route_logprob_names_the_sampled_token() -> None:
    """A logprob entry describes the sampled token, not the span it produced."""
    reply = HELD_STOP_START
    deltas = [dict(delta) for delta in reply.deltas]
    deltas[1]["top_logprobs"] = [
        {"token": " the E", "logprob": -0.25},
        {"token": " the", "logprob": -2.0},
    ]
    queue: asyncio.Queue = asyncio.Queue()
    for delta in deltas:
        await queue.put(delta)

    result = await asyncio.wait_for(
        chat_route.gather_non_streaming_batch_response(1, queue, _IpcStub(), [1], [1]),
        timeout=1.0,
    )

    (entry,) = result["choices"][0].logprobs.content
    assert entry.token == " the E"
    assert entry.logprob == -0.25


# The streaming route forwards ``content`` and never reads the spans, so it
# cannot repeat released text, and it shows whatever the tokens spelled,
# the stop sequence included.
_CHAT_STREAM_REPLIES = [
    pytest.param(
        reply,
        id=reply.name,
        marks=pytest.mark.xfail(
            strict=True,
            reason="the chat stream forwards content, which spells the stop sequence",
        )
        if reply is STOP_SEQUENCE
        else (),
    )
    for reply in ALL_REPLIES
]


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", _CHAT_STREAM_REPLIES)
async def test_chat_route_streaming(reply: Reply) -> None:
    chunks = [
        chunk["data"]
        async for chunk in chat_route.stream_response_generator(
            1, await _queue_of(reply), _IpcStub(), "test-model", 1
        )
    ]

    assert chunks[-1] == "[DONE]"
    text = "".join(
        choice["delta"].get("content") or ""
        for chunk in chunks[:-1]
        for choice in json.loads(chunk)["choices"]
    )
    assert text == reply.text


# --- orchard.clients ----------------------------------------------------------


@pytest.mark.parametrize("reply", ALL_REPLIES, ids=_ids(ALL_REPLIES))
def test_client_chat_aggregate(reply: Reply) -> None:
    assert Client._aggregate_message_text(_client_deltas(reply)) == reply.text


@pytest.mark.parametrize("reply", ALL_REPLIES, ids=_ids(ALL_REPLIES))
def test_client_responses_non_streaming(reply: Reply) -> None:
    response = aggregate_non_streaming_response(
        reply.deltas, "test-model", ResponsesRequest.from_text("hi")
    )

    assert _message_text(response.output) == reply.text


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", WITH_STATE_EVENTS, ids=_ids(WITH_STATE_EVENTS))
async def test_client_responses_streaming(reply: Reply) -> None:
    events = [
        event
        async for event in iter_response_events(_iterate(reply), model_id="test-model")
    ]

    streamed = "".join(
        event.delta for event in events if event.type == "response.output_text.delta"
    )
    final = next(
        event
        for event in events
        if event.type in ("response.completed", "response.incomplete")
    )
    assert streamed == reply.text
    assert _message_text(final.response.output) == reply.text


# --- /v1/responses --------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", WITH_STATE_EVENTS, ids=_ids(WITH_STATE_EVENTS))
async def test_responses_route_non_streaming(reply: Reply) -> None:
    result = await asyncio.wait_for(
        responses_route.gather_non_streaming_response(
            1, await _queue_of(reply), _IpcStub()
        ),
        timeout=1.0,
    )

    assert _message_text(result["output"]) == reply.text


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", WITH_STATE_EVENTS, ids=_ids(WITH_STATE_EVENTS))
async def test_responses_route_streaming(reply: Reply) -> None:
    events = [
        event
        async for event in responses_route.stream_response_generator(
            1, await _queue_of(reply), _IpcStub(), "resp_test", "test-model"
        )
    ]

    streamed = "".join(
        json.loads(event["data"])["delta"]
        for event in events
        if event.get("event") == "response.output_text.delta"
    )
    assert streamed == reply.text
