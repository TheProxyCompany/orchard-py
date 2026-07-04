import json

import pytest
from tests.golden.golden_io import assert_or_record
from tests.helpers import drain_stream, print_usage_summary, render_prompt_blue
from tests.models import Model

from orchard.clients.client import Client
from orchard.server.models.responses import (
    OutputFunctionCall,
    OutputStatus,
)

pytestmark = pytest.mark.asyncio


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


# The one correct tool, plus 17 plausible distractors. A capable model has to
# read intent and pick get_weather out of a crowded, overlapping toolbelt — not
# latch onto the first tool, the first location-shaped tool, or a near-synonym.
GET_WEATHER = _tool(
    "get_weather",
    "Get the current weather for a location.",
    {"location": {"type": "string"}},
    ["location"],
)

DISTRACTORS = [
    _tool("get_time", "Get the current time in a location.",
          {"location": {"type": "string"}}, ["location"]),
    _tool("get_news", "Get the latest news headlines for a topic.",
          {"topic": {"type": "string"}}, ["topic"]),
    _tool("send_email", "Send an email to a recipient.",
          {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}},
          ["to", "subject", "body"]),
    _tool("set_timer", "Set a countdown timer for a number of seconds.",
          {"seconds": {"type": "integer"}}, ["seconds"]),
    _tool("get_stock_price", "Get the current stock price for a ticker symbol.",
          {"ticker": {"type": "string"}}, ["ticker"]),
    _tool("translate_text", "Translate text into a target language.",
          {"text": {"type": "string"}, "target_language": {"type": "string"}},
          ["text", "target_language"]),
    _tool("get_directions", "Get driving directions between two locations.",
          {"origin": {"type": "string"}, "destination": {"type": "string"}},
          ["origin", "destination"]),
    _tool("create_calendar_event", "Create a calendar event.",
          {"title": {"type": "string"}, "start": {"type": "string"}}, ["title", "start"]),
    _tool("get_calendar_events", "List calendar events for a date.",
          {"date": {"type": "string"}}, ["date"]),
    _tool("play_music", "Play a song or playlist.",
          {"query": {"type": "string"}}, ["query"]),
    _tool("set_reminder", "Set a reminder at a given time.",
          {"text": {"type": "string"}, "time": {"type": "string"}}, ["text", "time"]),
    _tool("get_air_quality", "Get the air quality index for a location.",
          {"location": {"type": "string"}}, ["location"]),
    _tool("search_web", "Search the web for a query.",
          {"query": {"type": "string"}}, ["query"]),
    _tool("convert_currency", "Convert an amount between currencies.",
          {"amount": {"type": "number"}, "from": {"type": "string"}, "to": {"type": "string"}},
          ["amount", "from", "to"]),
    _tool("get_sports_score", "Get the latest score for a team.",
          {"team": {"type": "string"}}, ["team"]),
    _tool("book_flight", "Book a flight between two cities.",
          {"origin": {"type": "string"}, "destination": {"type": "string"}, "date": {"type": "string"}},
          ["origin", "destination", "date"]),
    _tool("get_traffic", "Get current traffic conditions for a location.",
          {"location": {"type": "string"}}, ["location"]),
]

TOOLS = [GET_WEATHER, *DISTRACTORS]

SYSTEM = (
    "You are a helpful assistant with tool calling. You have many tools "
    "available; select the single most appropriate one for the request."
)


async def test_tool_selection(client: Client, model: Model):
    """select the one right tool out of 18, streamed.

    Pins the streaming event lifecycle (exactly one function_call, opens with
    name known and empty args, closes completed) and the semantic outcome: out
    of get_weather plus 17 plausible distractors (get_time, get_air_quality,
    get_traffic, ... — several also take a `location`), the model picks
    get_weather with arguments {"location": "San Francisco"}, deterministically.
    """
    if not model.tools:
        return
    reasoning = {"effort": "medium"} if model.thinking else None
    print(
        f"\n\033[1;33m━━━ {model.template_type} · tool selection · 1-of-18 ━━━\033[0m",
        flush=True,
    )

    conversation = [
        {"type": "message", "role": "system", "content": SYSTEM},
        {"type": "message", "role": "user", "content": "What's the weather in San Francisco?"},
    ]

    # One turn: forced tool call; the model must select get_weather.
    turn1_request = dict(
        input=conversation,
        core_tools=TOOLS,
        tool_choice="required",
        deterministic=True,
        max_output_tokens=512,
        reasoning=reasoning,
        prefix_cache=False,
    )
    await render_prompt_blue(client, model.checkpoint, **turn1_request)
    stream = await client.aresponses(
        model.checkpoint, stream=True, stream_tokens=True, **turn1_request
    )
    turn1 = await drain_stream(stream)
    print_usage_summary([turn1])
    assert_or_record(model.template_type, "tool_selection", "turn1", turn1["events"])

    assert turn1["order"][0] == "response.created"
    assert turn1["order"][-1] == "done"
    assert turn1["counts"]["response.created"] == 1
    assert turn1["counts"]["response.in_progress"] == 1
    assert turn1["counts"]["response.completed"] == 1

    reasoning_blocks = turn1["added"].get("reasoning", 0)
    if reasoning_blocks:
        assert reasoning_blocks == 1, "turn1: expected at most one reasoning block"
        assert turn1["counts"]["response.reasoning.done"] == 1
        assert turn1["counts"]["response.reasoning.delta"] >= 1
        assert turn1["reasoning"].strip() == turn1["reasoning_done"], "turn1: reasoning deltas != reasoning.done"
    else:
        assert turn1["counts"].get("response.reasoning.delta", 0) == 0

    # a forced tool turn produces no assistant message text
    assert "response.output_text.delta" not in turn1["counts"], "turn1: leaked message text on a tool turn"

    # Exactly one tool call: not zero, not a fan-out across several tools.
    assert turn1["added"]["function_call"] == 1, "turn1: expected exactly one function_call opened"
    assert turn1["counts"]["response.function_call_arguments.done"] == 1, "turn1: expected one arguments.done"
    assert len(turn1["function_calls"]) == 1
    call = turn1["function_calls"][0]

    # The call opens with name + call_id known and empty arguments, streams args,
    # then closes completed — same lifecycle the desktop UI renders incrementally.
    opened = [item for item in turn1["items_added"] if isinstance(item, OutputFunctionCall)]
    assert len(opened) == 1, "turn1: expected one function_call opened"
    assert opened[0].name == "get_weather", (
        f"{model.template_type}: selected the wrong tool out of 18: {opened[0].name!r}"
    )
    assert opened[0].call_id == call.call_id
    assert opened[0].arguments == "", "turn1: function_call must open with empty arguments"
    assert opened[0].status == OutputStatus.IN_PROGRESS

    # The selection: get_weather, not any of the 17 distractors.
    distractor_names = {t["name"] for t in DISTRACTORS}
    assert call.name == "get_weather", (
        f"{model.template_type}: picked a distractor: {call.name!r}"
    )
    assert call.name not in distractor_names
    assert call.status == OutputStatus.COMPLETED
    assert json.loads(call.arguments) == {"location": "San Francisco"}, f"ACTUAL-ARGS: {call.arguments!r}"

    # Per-argument field_path tagging: value-only, format-agnostic.
    assert turn1["field_args"] == {"location": "San Francisco"}, (
        f"{model.template_type}: per-argument field_path tagging wrong "
        f"(expected location='San Francisco' value-only): {turn1['field_args']!r}"
    )
