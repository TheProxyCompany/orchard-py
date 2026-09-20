"""Message text is assembled once, from one view of the reply.

Every token delta carries two views of the text. ``content`` is the decoded
text of the sampled tokens. The state events are the engine's text stream:
it holds back the end of the text while that end could still become a stop
sequence (or the newline in front of ``</think>``), never contains a stop
sequence, and hands held text over in a later delta.

An engine that advertises the ``released_text`` capability completes that
stream: when a reply ends without a stop (token limit, cancel) the held text
arrives in the completion delta, which has no tokens and no ``content``.
Against such an engine a candidate that carries state events is assembled
from its message ``content_delta`` spans alone. Mixing in ``content`` repeats
text the spans deliver later ("Hello the EE") and shows text the engine held
back on purpose (the stop sequence, the markers of a tool call).

An engine without the capability never hands that held text over, so against
it every site assembles exactly as it did before the capability existed:
``/v1/chat/completions`` keeps reading ``content`` where it did (and with it
the stop sequence), the other sites keep reading the spans. A candidate that
never carries state events has only ``content``, whatever the engine.

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

from orchard.app.model_registry import ModelInfo
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


def _call_event(event_type: str, identifier: str, **fields: Any) -> dict[str, Any]:
    """An event of the tool call that follows the message, the second output item."""
    return {
        "event_type": event_type,
        "item_type": "tool_call",
        "output_index": 1,
        "identifier": identifier,
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
    # The deltas as an engine that advertises released_text sends them.
    deltas: list[dict[str, Any]]
    # The message text every assembly site makes of them.
    text: str
    # The tool calls of the reply, by name.
    calls: tuple[str, ...] = ()
    # What main made of the same reply from an engine without released_text,
    # where that is not ``text``: ``old_chat`` on /v1/chat/completions
    # (non-streaming), ``old_spans`` at the sites that read the spans alone.
    old_chat: str | None = None
    old_spans: str | None = None


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
    # The held "E" never arrives: only content has it.
    old_spans="Hello the ",
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
    # The held newline never arrives, and the chat route took content only
    # from a delta with spans.
    old_chat="Hello",
    old_spans="Hello",
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

# The stop sequence "END" inside a token: the span ends in front of it,
# content spells it.
STOP_INSIDE_TOKEN = Reply(
    "stop_sequence_inside_a_token",
    [
        _HELLO,
        _token(
            " END",
            _event("content_delta", delta=" "),
            _event("item_completed", value="Hello "),
        ),
        _final("stop"),
    ],
    "Hello ",
    # The chat route took content where it extended the spans.
    old_chat="Hello END",
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
    # The chat route took content where it extended the spans: " the" for the
    # span " th", then the span "e cat".
    old_chat="Hello thee cat",
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

# The reply ends on a stop token while a newline is held: the text state
# machine ends with the reply, so the completion delta carries the newline and
# the finished message.
STOP_TOKEN_HELD_NEWLINE = Reply(
    "stop_token_releases_held_newline",
    [
        _HELLO,
        _token("\n"),
        _final(
            "stop",
            _event("content_delta", delta="\n"),
            _event("item_completed", value="Hello\n"),
        ),
    ],
    "Hello\n",
)

_ARGUMENTS = '{"city":"Paris"}'
_CALL_EVENTS = (
    _call_event("item_started", "tool_call:get_weather"),
    _call_event("content_delta", "arguments", delta=_ARGUMENTS),
    _call_event(
        "item_completed",
        "tool_call:get_weather",
        value=json.dumps({"name": "get_weather", "arguments": {"city": "Paris"}}),
    ),
)

# A tool call after held text: the opening marker ends the message, so its
# delta releases the held newline, completes the message and starts the call.
# The markers are tokens like any other and are decoded into content.
TOOL_CALL_AFTER_HELD_TEXT = Reply(
    "tool_call_after_held_text",
    [
        _HELLO,
        _token("\n"),
        _token(
            "<tool_call>",
            _event("content_delta", delta="\n"),
            _event("item_completed", value="Hello\n"),
            _CALL_EVENTS[0],
        ),
        _token(_ARGUMENTS, _CALL_EVENTS[1]),
        _token("</tool_call>", _CALL_EVENTS[2]),
        _final("tool_use"),
    ],
    "Hello\n",
    calls=("get_weather",),
)

# The markers of a tool call as deltas without events, after the message item
# exists: they are not message text.
MARKERS_AFTER_MESSAGE = Reply(
    "markers_without_events_after_a_message",
    [
        _HELLO,
        _token("<tool_call>"),
        _token(_ARGUMENTS, _event("item_completed", value="Hello"), *_CALL_EVENTS),
        _token("</tool_call>"),
        _final("tool_use"),
    ],
    "Hello",
    calls=("get_weather",),
)

# Cancelled by the client (the engine's finish reason is "user") after " the E":
# like the token limit, a cancel ends the reply without a stop, and the
# completion delta releases the held "E".
CANCELLED = Reply(
    "cancel_releases_held_stop_start",
    [
        _HELLO,
        _token(" the E", _event("content_delta", delta=" the ")),
        _final("user", _event("content_delta", delta="E")),
    ],
    "Hello the E",
    old_spans="Hello the ",
)

_JSON_START = _token(
    '{"answer":', _event("item_started"), _event("content_delta", delta='{"answer":')
)

# Structured output: the spans carry the JSON and the finished item repeats it
# as its value.
JSON_COMPLETE = Reply(
    "structured_json_complete",
    [
        _JSON_START,
        _token(
            '"A"}',
            _event("content_delta", delta='"A"}'),
            _event("item_completed", value='{"answer":"A"}'),
        ),
        _final("stop"),
    ],
    '{"answer":"A"}',
)

# Structured output cut by the token limit in the middle of the JSON.
JSON_CUT = Reply(
    "structured_json_cut_by_length",
    [
        _JSON_START,
        _token('"A', _event("content_delta", delta='"A')),
        _final("length"),
    ],
    '{"answer":"A',
)

ALL_REPLIES = [
    HELD_STOP_START,
    HELD_NEWLINE,
    STOP_SEQUENCE,
    STOP_INSIDE_TOKEN,
    HELD_MID_REPLY,
    NO_STATE_EVENTS,
    STOP_TOKEN_HELD_NEWLINE,
    TOOL_CALL_AFTER_HELD_TEXT,
    MARKERS_AFTER_MESSAGE,
    CANCELLED,
    JSON_COMPLETE,
    JSON_CUT,
]


@dataclass(frozen=True)
class Case:
    """A recorded reply from one of the two engines, and what the sites make of it."""

    name: str
    released_text: bool
    deltas: list[dict[str, Any]]
    chat: str  # /v1/chat/completions, non-streaming
    chat_stream: str  # /v1/chat/completions, stream
    spans: str  # orchard.clients and /v1/responses, which always read the spans
    calls: tuple[str, ...]


def _from_old_engine(reply: Reply) -> Case:
    """The reply as an engine without released_text sends it, as main assembled it.

    Such an engine hands held text over only while the reply goes on: the
    completion delta of a reply that ended without a stop carries no events.
    """
    deltas = [
        {**delta, "state_events": []}
        if delta["is_final_delta"] and delta["finish_reason"] in ("length", "user")
        else delta
        for delta in reply.deltas
    ]
    return Case(
        name=f"{reply.name}-old_engine",
        released_text=False,
        deltas=deltas,
        chat=reply.text if reply.old_chat is None else reply.old_chat,
        # The stream forwarded content as it came.
        chat_stream="".join(delta["content"] for delta in deltas),
        spans=reply.text if reply.old_spans is None else reply.old_spans,
        calls=reply.calls,
    )


CASES = [
    case
    for reply in ALL_REPLIES
    for case in (
        Case(
            name=f"{reply.name}-released_text",
            released_text=True,
            deltas=reply.deltas,
            chat=reply.text,
            chat_stream=reply.text,
            spans=reply.text,
            calls=reply.calls,
        ),
        _from_old_engine(reply),
    )
]

all_cases = pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])


class _IpcStub:
    async def next_delta(self, queue: asyncio.Queue) -> dict:
        return await queue.get()


async def _queue_of(deltas: list[dict[str, Any]]) -> asyncio.Queue:
    queue: asyncio.Queue = asyncio.Queue()
    for delta in deltas:
        await queue.put(dict(delta))
    return queue


async def _iterate(deltas: list[dict[str, Any]]):
    for delta in deltas:
        yield delta


def _message_text(output: list[Any]) -> str:
    return "".join(
        part.text for item in output if item.type == "message" for part in item.content
    )


def _calls(output: list[Any]) -> tuple[str, ...]:
    return tuple(item.name for item in output if item.type == "function_call")


# --- /v1/chat/completions ---------------------------------------------------


async def _chat_route_choice(deltas: list[dict[str, Any]], released_text: bool) -> Any:
    result = await asyncio.wait_for(
        chat_route.gather_non_streaming_batch_response(
            1,
            await _queue_of(deltas),
            _IpcStub(),
            [1],
            [1],
            released_text=released_text,
        ),
        timeout=1.0,
    )
    return result["choices"][0]


@pytest.mark.asyncio
@all_cases
async def test_chat_route_non_streaming(case: Case) -> None:
    choice = await _chat_route_choice(case.deltas, case.released_text)

    assert choice.message.content == case.chat


_MID_REPLY_LOGPROBS = [
    {"token": " cat", "logprob": -0.25},
    {"token": " dog", "logprob": -2.0},
]


@pytest.mark.asyncio
async def test_chat_route_logprob_names_the_sampled_token() -> None:
    """A logprob entry describes the sampled token, not the span it produced."""
    deltas = [dict(delta) for delta in HELD_MID_REPLY.deltas]
    deltas[2]["top_logprobs"] = _MID_REPLY_LOGPROBS

    choice = await _chat_route_choice(deltas, released_text=True)

    (entry,) = choice.logprobs.content
    assert (entry.token, entry.logprob) == (" cat", -0.25)


@pytest.mark.asyncio
async def test_chat_route_logprob_entry_is_unchanged_for_an_old_engine() -> None:
    """Main named the entry after the text it appended, here the span "e cat"."""
    deltas = [dict(delta) for delta in HELD_MID_REPLY.deltas]
    deltas[2]["top_logprobs"] = _MID_REPLY_LOGPROBS

    choice = await _chat_route_choice(deltas, released_text=False)

    (entry,) = choice.logprobs.content
    assert (entry.token, entry.logprob) == ("e cat", -999.0)


@pytest.mark.asyncio
@all_cases
async def test_chat_route_streaming(case: Case) -> None:
    chunks = [
        chunk["data"]
        async for chunk in chat_route.stream_response_generator(
            1,
            await _queue_of(case.deltas),
            _IpcStub(),
            "test-model",
            1,
            released_text=case.released_text,
        )
    ]

    assert chunks[-1] == "[DONE]"
    text = "".join(
        choice["delta"].get("content") or ""
        for chunk in chunks[:-1]
        for choice in json.loads(chunk)["choices"]
    )
    assert text == case.chat_stream


async def _both_chat_routes(deltas: list[dict[str, Any]]) -> tuple[str, str]:
    """The streamed and the non-streaming text, against an engine with released_text."""
    chunks = [
        chunk["data"]
        async for chunk in chat_route.stream_response_generator(
            1, await _queue_of(deltas), _IpcStub(), "test-model", 1, released_text=True
        )
    ]
    streamed = "".join(
        choice["delta"].get("content") or ""
        for chunk in chunks[:-1]
        for choice in json.loads(chunk)["choices"]
    )
    choice = await _chat_route_choice(deltas, released_text=True)
    return streamed, choice.message.content


@pytest.mark.asyncio
async def test_chat_route_streaming_leaves_reasoning_out() -> None:
    """The stream agrees with the non-streaming route, which returns no reasoning text."""
    reasoning = {"item_type": "reasoning", "identifier": "reasoning"}
    deltas = [
        _token("<think>"),
        _token(
            "hm",
            _event("item_started", **reasoning),
            _event("content_delta", delta="hm", **reasoning),
        ),
        _token("</think>", _event("item_completed", value="hm", **reasoning)),
        _HELLO,
        _final("stop"),
    ]

    assert await _both_chat_routes(deltas) == ("Hello", "Hello")


@pytest.mark.asyncio
async def test_chat_route_streaming_shows_a_message_that_arrives_as_a_value() -> None:
    """A message without spans is the value of its completion, on both routes."""
    deltas = [
        _token('{"answer":', _event("item_started")),
        _token('"A"}', _event("item_completed", value='{"answer":"A"}')),
        _final("stop"),
    ]

    assert await _both_chat_routes(deltas) == ('{"answer":"A"}', '{"answer":"A"}')


@pytest.mark.asyncio
async def test_chat_route_streaming_keeps_candidates_apart() -> None:
    """One candidate's state events say nothing about another's content."""
    with_events = [{**delta, "sequence_id": 7} for delta in STOP_SEQUENCE.deltas]
    without_events = [
        {**delta, "candidate_index": 1, "sequence_id": 8}
        for delta in NO_STATE_EVENTS.deltas
    ]
    deltas = [
        delta
        for pair in zip(with_events, [*without_events, None], strict=True)
        for delta in pair
        if delta is not None
    ]

    chunks = [
        chunk["data"]
        async for chunk in chat_route.stream_response_generator(
            1, await _queue_of(deltas), _IpcStub(), "test-model", 2, released_text=True
        )
    ]
    text = {0: "", 1: ""}
    for chunk in chunks[:-1]:
        for choice in json.loads(chunk)["choices"]:
            text[choice["index"]] += choice["delta"].get("content") or ""

    assert text == {0: STOP_SEQUENCE.text, 1: NO_STATE_EVENTS.text}


# --- orchard.clients ----------------------------------------------------------


@all_cases
def test_client_chat_aggregate(case: Case) -> None:
    deltas = [ClientDelta(**delta) for delta in case.deltas]
    _, tool_calls = Client._aggregate_structured_items(deltas)

    assert Client._aggregate_message_text(deltas) == case.spans
    assert tuple(call["name"] for call in tool_calls) == case.calls


@all_cases
def test_client_responses_non_streaming(case: Case) -> None:
    response = aggregate_non_streaming_response(
        case.deltas, "test-model", ResponsesRequest.from_text("hi")
    )

    assert _message_text(response.output) == case.spans
    assert _calls(response.output) == case.calls


@pytest.mark.asyncio
@all_cases
async def test_client_responses_streaming(case: Case) -> None:
    events = [
        event
        async for event in iter_response_events(
            _iterate(case.deltas), model_id="test-model"
        )
    ]

    streamed = "".join(
        event.delta for event in events if event.type == "response.output_text.delta"
    )
    final = next(
        event
        for event in events
        if event.type in ("response.completed", "response.incomplete")
    )
    assert streamed == case.spans
    assert _message_text(final.response.output) == case.spans
    assert _calls(final.response.output) == case.calls


# --- /v1/responses --------------------------------------------------------------


@pytest.mark.asyncio
@all_cases
async def test_responses_route_non_streaming(case: Case) -> None:
    result = await asyncio.wait_for(
        responses_route.gather_non_streaming_response(
            1, await _queue_of(case.deltas), _IpcStub()
        ),
        timeout=1.0,
    )

    assert _message_text(result["output"]) == case.spans
    assert _calls(result["output"]) == case.calls


@pytest.mark.asyncio
@all_cases
async def test_responses_route_streaming(case: Case) -> None:
    events = [
        event
        async for event in responses_route.stream_response_generator(
            1, await _queue_of(case.deltas), _IpcStub(), "resp_test", "test-model"
        )
    ]

    streamed = "".join(
        json.loads(event["data"])["delta"]
        for event in events
        if event.get("event") == "response.output_text.delta"
    )
    final = next(
        json.loads(event["data"])["response"]["output"]
        for event in events
        if event.get("event") in ("response.completed", "response.incomplete")
    )
    assert streamed == case.spans
    assert [
        part["text"]
        for item in final
        if item["type"] == "message"
        for part in item["content"]
    ] == ([case.spans] if case.spans else [])
    assert (
        tuple(item["name"] for item in final if item["type"] == "function_call")
        == case.calls
    )


# --- the capability ---------------------------------------------------------------


def test_model_info_reads_the_released_text_capability() -> None:
    def info(capabilities: dict[str, list[int]] | None) -> ModelInfo:
        return ModelInfo("model", "/models/model", None, capabilities)  # type: ignore[arg-type]

    assert info({"released_text": [1], "answer": [7]}).releases_held_text
    assert not info({"answer": [7]}).releases_held_text
    assert not info(None).releases_held_text
