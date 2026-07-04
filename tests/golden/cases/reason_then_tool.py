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

SYSTEM = (
    "You are a helpful assistant with tool calling. Reason about the request, "
    "then call a tool when needed and use its result to answer."
)


async def test_reason_then_tool(client: Client, model: Model):
    """reason -> tool call -> tool result -> grounded answer, streamed.

    Pins the streaming event lifecycle (exactly-once reasoning block, delta
    accumulation, one tool call) and the semantic outcome (right tool, right
    args, answer grounded in the result), deterministically.
    """
    if not model.tools:
        return
    reasoning = {"effort": "medium"} if model.thinking else None
    print(
        f"\n\033[1;33m━━━ {model.template_type} · reason → tool · multi-turn ━━━\033[0m",
        flush=True,
    )

    conversation = [
        {"type": "message", "role": "system", "content": SYSTEM},
        {"type": "message", "role": "user", "content": "What's the weather in San Francisco?"},
    ]

    # Turn 1: reason, then call get_weather(location="San Francisco").
    turn1_request = dict(
        input=conversation,
        core_tools=[GET_WEATHER],
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
    assert_or_record(model.template_type, "reason_then_tool", "turn1", turn1["events"])

    assert turn1["order"][0] == "response.created"
    assert turn1["order"][-1] == "done"
    assert turn1["counts"]["response.created"] == 1
    assert turn1["counts"]["response.in_progress"] == 1
    assert turn1["counts"]["response.completed"] == 1

    turn1_reasoning_blocks = turn1["added"].get("reasoning", 0)
    if not model.thinking:
        assert turn1_reasoning_blocks == 0, "turn1: non-reasoning model reasoned"
    if turn1_reasoning_blocks:
        assert turn1_reasoning_blocks == 1, "turn1: expected exactly one reasoning block"
        assert turn1["counts"]["response.reasoning.done"] == 1
        assert turn1["counts"]["response.reasoning.delta"] >= 1
        assert turn1["reasoning"].strip() == turn1["reasoning_done"], "turn1: reasoning deltas != reasoning.done"
    else:
        assert turn1["counts"].get("response.reasoning.delta", 0) == 0

    # a tool turn produces no assistant message text
    assert "response.output_text.delta" not in turn1["counts"], "turn1: leaked message text on a tool turn"

    assert turn1["added"]["function_call"] == 1, "turn1: expected exactly one function_call opened"
    assert turn1["counts"]["response.function_call_arguments.done"] == 1, "turn1: expected one arguments.done"
    assert len(turn1["function_calls"]) == 1
    call = turn1["function_calls"][0]

    # The desktop UI renders the tool call incrementally off this stream, so pin the
    # lifecycle it binds to: name + call_id are known when the item OPENS (args still
    # empty), arguments stream as deltas, and the DONE item carries the full normalized
    # arguments. (We don't pin the raw delta byte form — that's per-model tokenization.)
    opened = [item for item in turn1["items_added"] if isinstance(item, OutputFunctionCall)]
    assert len(opened) == 1, "turn1: expected one function_call opened"
    assert opened[0].name == "get_weather"
    assert opened[0].call_id == call.call_id
    assert opened[0].arguments == "", "turn1: function_call must open with empty arguments"
    assert opened[0].status == OutputStatus.IN_PROGRESS
    assert call.name == "get_weather"
    assert call.status == OutputStatus.COMPLETED
    assert json.loads(call.arguments) == {"location": "San Francisco"}, f"ACTUAL-ARGS: {call.arguments!r}"

    # Per-argument field_path tagging. The desktop UI binds each argument to its own
    # field_path to render the call incrementally, so every native tool format must
    # decompose the same way: value chunks carry the argument name, structural
    # boilerplate (pythonic '(location=' / ')', json punctuation) stays untagged and is
    # excluded from field_args. Value-only, format-agnostic. A model that streams the
    # whole blob untagged is a tagging regression in PSE/PIE, not a model quirk.
    assert turn1["field_args"] == {"location": "San Francisco"}, (
        f"{model.template_type}: per-argument field_path tagging wrong "
        f"(expected location='San Francisco' value-only): {turn1['field_args']!r}"
    )

    # Turn 2: feed the tool result back; the model answers using it.
    conversation += [
        {"type": "function_call", "call_id": call.call_id, "name": call.name, "arguments": turn1["args_done"]},
        {
            "type": "function_call_output",
            "call_id": call.call_id,
            "output": json.dumps({"temperature": 65, "unit": "fahrenheit", "condition": "foggy"}),
        },
    ]
    turn2_request = dict(
        input=conversation,
        core_tools=[GET_WEATHER],
        tool_choice="none",
        deterministic=True,
        max_output_tokens=512,
        reasoning=reasoning,
        prefix_cache=False,
    )
    gen1 = turn1["generated"] + (turn1["stop_token"] or "")
    await render_prompt_blue(
        client, model.checkpoint, prev_gen=gen1, **turn2_request
    )
    stream = await client.aresponses(
        model.checkpoint, stream=True, stream_tokens=True, **turn2_request
    )
    turn2 = await drain_stream(stream)
    print_usage_summary([turn1, turn2])
    assert_or_record(model.template_type, "reason_then_tool", "turn2", turn2["events"])

    assert turn2["order"][0] == "response.created"
    assert turn2["order"][-1] == "done"
    assert turn2["counts"]["response.created"] == 1
    assert turn2["counts"]["response.in_progress"] == 1
    assert turn2["counts"]["response.completed"] == 1

    turn2_reasoning_blocks = turn2["added"].get("reasoning", 0)
    if not model.thinking:
        assert turn2_reasoning_blocks == 0, "turn2: non-reasoning model reasoned"
    if turn2_reasoning_blocks:
        assert turn2_reasoning_blocks == 1, "turn2: expected exactly one reasoning block"
        assert turn2["counts"]["response.reasoning.done"] == 1
        assert turn2["counts"]["response.reasoning.delta"] >= 1
        assert turn2["reasoning"].strip() == turn2["reasoning_done"], "turn2: reasoning deltas != reasoning.done"
        assert "<|" not in turn2["reasoning"] and "</" not in turn2["reasoning"], "turn2: control leak in reasoning"
    else:
        assert turn2["counts"].get("response.reasoning.delta", 0) == 0

    assert turn2["counts"].get("response.function_call_arguments.done", 0) == 0, "turn2: unexpected tool call"
    assert turn2["counts"]["response.output_text.done"] == 1, "turn2: expected one message"
    assert turn2["content"] == turn2["content_done"], "turn2: content deltas != output_text.done"

    # message lifecycle the UI streams: opens as an empty assistant message, fills via
    # output_text deltas, closes completed.
    msg_open = [item for item in turn2["items_added"] if isinstance(item, OutputMessage)]
    msg_done = [item for item in turn2["items_done"] if isinstance(item, OutputMessage)]
    assert len(msg_open) == 1 and len(msg_done) == 1, "turn2: expected one message item"
    assert msg_open[0].role == "assistant"
    assert not msg_open[0].content, "turn2: message must open with empty content"
    assert msg_open[0].status == OutputStatus.IN_PROGRESS
    assert msg_done[0].status == OutputStatus.COMPLETED

    answer = turn2["content_done"].lower()
    assert "65" in answer or "fog" in answer, f"{model.template_type}: answer ignored the tool result: {answer!r}"
