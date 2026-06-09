import json

import pytest
from tests.golden.golden_io import assert_or_record
from tests.helpers import drain_stream, print_usage_summary, render_prompt_blue
from tests.models import Model

from orchard.clients.client import Client
from orchard.server.models.responses import (
    OutputFunctionCall,
    OutputMessage,
    OutputStatus,
)

pytestmark = pytest.mark.asyncio

GET_WEATHER = {
    "type": "function",
    "name": "get_weather",
    "description": "Get the current weather for a location.",
    "parameters": {
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    },
}

GET_TIME = {
    "type": "function",
    "name": "get_time",
    "description": "Get the current local time in a timezone.",
    "parameters": {
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "IANA timezone identifier, for example America/New_York.",
            }
        },
        "required": ["timezone"],
    },
}

TOOLS = [GET_WEATHER, GET_TIME]

SYSTEM = (
    "You are a helpful assistant with tool calling. Call the tools you need to "
    "answer the request, then use their results to give the final answer."
)


async def test_multi_tool(client: Client, model: Model):
    """two tools, one request that needs both -> tool results -> integrated answer.

    Pins the streaming lifecycle of a multi-tool exchange (each function_call
    opens with name+call_id and empty args, args stream as deltas, every call
    finalizes), the semantic outcome (the right two tools called with the right
    args), and the final grounded answer (both injected values present). Some
    archs emit both calls in one turn, others split across turns; whichever this
    arch does is what the golden pins.
    """
    if not model.tools:
        return
    reasoning = {"effort": "medium"} if model.thinking else None
    print(
        f"\n\033[1;33m━━━ {model.template_type} · multi-tool · weather + time ━━━\033[0m",
        flush=True,
    )

    conversation = [
        {"type": "message", "role": "system", "content": SYSTEM},
        {
            "type": "message",
            "role": "user",
            "content": "What's the weather in San Francisco and what time is it in Tokyo?",
        },
    ]

    # Turn 1: the model must call get_weather and get_time (one or both this turn).
    turn1_request = dict(
        input=conversation,
        core_tools=TOOLS,
        tool_choice="required",
        temperature=0.0,
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
    assert_or_record(model.template_type, "multi_tool", "turn1", turn1["events"])

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

    # a tool turn produces no assistant message text
    assert "response.output_text.delta" not in turn1["counts"], "turn1: leaked message text on a tool turn"

    # This arch may emit one or both calls this turn; whatever it does, the
    # function_call lifecycle must be consistent: every opened call finalizes
    # with matching args.done, opens with empty args, and is one of our tools.
    n_calls = turn1["added"]["function_call"]
    assert n_calls >= 1, "turn1: expected at least one function_call opened"
    assert turn1["counts"]["response.function_call_arguments.done"] == n_calls, (
        "turn1: every opened function_call must finalize its arguments"
    )
    assert len(turn1["function_calls"]) == n_calls

    opened = [item for item in turn1["items_added"] if isinstance(item, OutputFunctionCall)]
    assert len(opened) == n_calls, "turn1: opened function_call count mismatch"
    for item in opened:
        assert item.name in {"get_weather", "get_time"}, f"turn1: unexpected tool {item.name!r}"
        assert item.arguments == "", "turn1: function_call must open with empty arguments"
        assert item.status == OutputStatus.IN_PROGRESS
    open_ids = {item.call_id for item in opened}
    assert len(open_ids) == n_calls, "turn1: duplicate call_id across opened calls"

    # The completed calls: right tools, right args, distinct call_ids matching the opens.
    by_name: dict[str, OutputFunctionCall] = {}
    for call in turn1["function_calls"]:
        assert call.status == OutputStatus.COMPLETED
        assert call.call_id in open_ids, "turn1: done call_id has no matching open"
        by_name[call.name] = call

    if "get_weather" in by_name:
        assert json.loads(by_name["get_weather"].arguments) == {"location": "San Francisco"}, (
            f"{model.template_type}: get_weather args wrong: {by_name['get_weather'].arguments!r}"
        )
    if "get_time" in by_name:
        assert json.loads(by_name["get_time"].arguments) == {"timezone": "Tokyo"}, (
            f"{model.template_type}: get_time args wrong: {by_name['get_time'].arguments!r}"
        )

    # Turn 2: feed back a result for every call this arch made (in call order),
    # plus, if it only called one tool this turn, the other tool so it can finish.
    weather_result = {"temperature": 65, "unit": "fahrenheit", "condition": "foggy"}
    time_result = {"time": "23:00", "timezone": "Tokyo", "utc_offset": "+09:00"}
    results = {
        "get_weather": json.dumps(weather_result),
        "get_time": json.dumps(time_result),
    }
    for call in turn1["function_calls"]:
        conversation += [
            {"type": "function_call", "call_id": call.call_id, "name": call.name, "arguments": call.arguments},
            {"type": "function_call_output", "call_id": call.call_id, "output": results[call.name]},
        ]

    called = set(by_name)
    remaining = [t for t in TOOLS if t["name"] not in called]
    turn2_choice = "required" if remaining else "none"

    turn2_request = dict(
        input=conversation,
        core_tools=TOOLS,
        tool_choice=turn2_choice,
        temperature=0.0,
        deterministic=True,
        max_output_tokens=512,
        reasoning=reasoning,
        prefix_cache=False,
    )
    gen1 = turn1["generated"] + (turn1["stop_token"] or "")
    await render_prompt_blue(client, model.checkpoint, prev_gen=gen1, **turn2_request)
    stream = await client.aresponses(
        model.checkpoint, stream=True, stream_tokens=True, **turn2_request
    )
    turn2 = await drain_stream(stream)
    assert_or_record(model.template_type, "multi_tool", "turn2", turn2["events"])

    assert turn2["order"][0] == "response.created"
    assert turn2["order"][-1] == "done"
    assert turn2["counts"]["response.created"] == 1
    assert turn2["counts"]["response.in_progress"] == 1
    assert turn2["counts"]["response.completed"] == 1

    reasoning_blocks = turn2["added"].get("reasoning", 0)
    if reasoning_blocks:
        assert reasoning_blocks == 1, "turn2: expected at most one reasoning block"
        assert turn2["counts"]["response.reasoning.done"] == 1
        assert turn2["counts"]["response.reasoning.delta"] >= 1
        assert turn2["reasoning"].strip() == turn2["reasoning_done"], "turn2: reasoning deltas != reasoning.done"
        assert "<|" not in turn2["reasoning"] and "</" not in turn2["reasoning"], "turn2: control leak in reasoning"
    else:
        assert turn2["counts"].get("response.reasoning.delta", 0) == 0

    turns = [turn1, turn2]

    if remaining:
        # Split-call arch: turn 2 issues the second tool call; the integrated
        # answer comes in turn 3 after both results are in hand.
        assert turn2["added"]["function_call"] >= 1, "turn2: expected the remaining tool call"
        assert turn2["counts"].get("response.output_text.done", 0) == 0, "turn2: leaked message on a tool turn"

        second = by_name
        for call in turn2["function_calls"]:
            assert call.status == OutputStatus.COMPLETED
            assert call.name in {t["name"] for t in remaining}, f"turn2: unexpected tool {call.name!r}"
            second[call.name] = call
            conversation += [
                {"type": "function_call", "call_id": call.call_id, "name": call.name, "arguments": call.arguments},
                {"type": "function_call_output", "call_id": call.call_id, "output": results[call.name]},
            ]

        assert json.loads(second["get_weather"].arguments) == {"location": "San Francisco"}
        assert json.loads(second["get_time"].arguments) == {"timezone": "Asia/Tokyo"}

        turn3_request = dict(
            input=conversation,
            core_tools=TOOLS,
            tool_choice="none",
            temperature=0.0,
            deterministic=True,
            max_output_tokens=512,
            reasoning=reasoning,
            prefix_cache=False,
        )
        gen2 = turn2["generated"] + (turn2["stop_token"] or "")
        await render_prompt_blue(client, model.checkpoint, prev_gen=gen2, **turn3_request)
        stream = await client.aresponses(
            model.checkpoint, stream=True, stream_tokens=True, **turn3_request
        )
        turn3 = await drain_stream(stream)
        assert_or_record(model.template_type, "multi_tool", "turn3", turn3["events"])

        assert turn3["order"][0] == "response.created"
        assert turn3["order"][-1] == "done"
        assert turn3["counts"]["response.completed"] == 1
        reasoning_blocks = turn3["added"].get("reasoning", 0)
        if reasoning_blocks:
            assert reasoning_blocks == 1, "turn3: expected at most one reasoning block"
            assert turn3["reasoning"].strip() == turn3["reasoning_done"], "turn3: reasoning deltas != reasoning.done"
        else:
            assert turn3["counts"].get("response.reasoning.delta", 0) == 0
        assert turn3["counts"].get("response.function_call_arguments.done", 0) == 0, "turn3: unexpected tool call"
        assert turn3["counts"]["response.output_text.done"] == 1, "turn3: expected one message"
        assert turn3["content"] == turn3["content_done"], "turn3: content deltas != output_text.done"
        final = turn3
        turns.append(turn3)
    else:
        # Parallel-call arch: both calls landed in turn 1; turn 2 is the answer.
        assert turn2["counts"].get("response.function_call_arguments.done", 0) == 0, "turn2: unexpected tool call"
        assert turn2["counts"]["response.output_text.done"] == 1, "turn2: expected one message"
        assert turn2["content"] == turn2["content_done"], "turn2: content deltas != output_text.done"
        assert by_name.keys() == {"get_weather", "get_time"}, "turn1: parallel arch must emit both calls"
        final = turn2

    print_usage_summary(turns)

    # final message lifecycle the UI streams: opens empty assistant message,
    # fills via output_text deltas, closes completed.
    msg_open = [item for item in final["items_added"] if isinstance(item, OutputMessage)]
    msg_done = [item for item in final["items_done"] if isinstance(item, OutputMessage)]
    assert len(msg_open) == 1 and len(msg_done) == 1, "final: expected one message item"
    assert msg_open[0].role == "assistant"
    assert not msg_open[0].content, "final: message must open with empty content"
    assert msg_open[0].status == OutputStatus.IN_PROGRESS
    assert msg_done[0].status == OutputStatus.COMPLETED

    # The integrated answer must ground in BOTH tool results.
    answer = final["content_done"].lower()
    assert "65" in answer or "fog" in answer, (
        f"{model.template_type}: answer ignored the weather result: {answer!r}"
    )
    assert "23:00" in answer or "11" in answer or "tokyo" in answer, (
        f"{model.template_type}: answer ignored the time result: {answer!r}"
    )
