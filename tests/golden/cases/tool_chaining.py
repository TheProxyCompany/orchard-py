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

FIND_KEY = {
    "type": "function",
    "name": "find_key",
    "description": "Search a room and return the key hidden there.",
    "parameters": {
        "type": "object",
        "properties": {"room": {"type": "string"}},
        "required": ["room"],
    },
}

UNLOCK_CHEST = {
    "type": "function",
    "name": "unlock_chest",
    "description": "Unlock the treasure chest with a key and return its contents.",
    "parameters": {
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
    },
}

TOOLS = [FIND_KEY, UNLOCK_CHEST]

# Unguessable values: the only way to pass is to chain turn-1's output into
# turn-2's input, and to ground the answer in turn-2's result.
KEY = "K7-MAGENTA-9931"
CHEST_CONTENTS = "a jade dragon figurine"

SYSTEM = (
    "You are a helpful assistant with tool calling. Use the tools in the right "
    "order, passing each tool's result into the next, then answer the request."
)


async def test_tool_chaining(client: Client, model: Model):
    """sequential, dependent tools: find_key -> unlock_chest(that key) -> answer.

    Pins the data-flow contract a parallel multi-tool test can't: the model must
    call find_key FIRST (it can't unlock without a key), then feed the EXACT key
    that tool returned into unlock_chest(key=...), then ground the answer in the
    chest contents. The key is unguessable, so a hallucinated key or wrong tool
    order fails. Per turn it also pins the streaming lifecycle and field_path.
    """
    if not model.tools:
        return
    reasoning = {"effort": "medium"} if model.thinking else None
    print(
        f"\n\033[1;33m━━━ {model.template_type} · tool chaining · key → chest ━━━\033[0m",
        flush=True,
    )

    conversation = [
        {"type": "message", "role": "system", "content": SYSTEM},
        {
            "type": "message",
            "role": "user",
            "content": "Find the key hidden in the library, then unlock the treasure chest with it and tell me what's inside.",
        },
    ]

    # Turn 1: the model must call find_key first (it has no key yet to unlock).
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
    assert_or_record(model.template_type, "tool_chaining", "turn1", turn1["events"])

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

    assert "response.output_text.delta" not in turn1["counts"], "turn1: leaked message text on a tool turn"

    # Exactly one call, and it must be find_key — calling unlock_chest now (no key
    # in hand) would be the wrong order and a hallucinated key.
    assert turn1["added"]["function_call"] == 1, "turn1: expected exactly one function_call"
    assert turn1["counts"]["response.function_call_arguments.done"] == 1
    assert len(turn1["function_calls"]) == 1
    find = turn1["function_calls"][0]
    assert find.name == "find_key", f"turn1: must call find_key first, got {find.name!r}"
    assert find.status == OutputStatus.COMPLETED
    assert "library" in json.loads(find.arguments).get("room", "").lower(), (
        f"turn1: find_key should search the library: {find.arguments!r}"
    )

    # Turn 2: feed back the unguessable key; the model must unlock with THAT key.
    conversation += [
        {"type": "function_call", "call_id": find.call_id, "name": find.name, "arguments": find.arguments},
        {"type": "function_call_output", "call_id": find.call_id, "output": json.dumps({"key": KEY})},
    ]
    turn2_request = dict(
        input=conversation,
        core_tools=TOOLS,
        tool_choice="required",
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
    assert_or_record(model.template_type, "tool_chaining", "turn2", turn2["events"])

    assert turn2["order"][0] == "response.created"
    assert turn2["order"][-1] == "done"
    assert turn2["counts"]["response.completed"] == 1

    reasoning_blocks = turn2["added"].get("reasoning", 0)
    if reasoning_blocks:
        assert reasoning_blocks == 1, "turn2: expected at most one reasoning block"
        assert turn2["counts"]["response.reasoning.done"] == 1
        assert turn2["reasoning"].strip() == turn2["reasoning_done"], "turn2: reasoning deltas != reasoning.done"
        assert "<|" not in turn2["reasoning"] and "</" not in turn2["reasoning"], "turn2: control leak in reasoning"
    else:
        assert turn2["counts"].get("response.reasoning.delta", 0) == 0

    assert "response.output_text.delta" not in turn2["counts"], "turn2: leaked message text on a tool turn"

    assert turn2["added"]["function_call"] == 1, "turn2: expected exactly one function_call"
    assert len(turn2["function_calls"]) == 1
    unlock = turn2["function_calls"][0]
    assert unlock.name == "unlock_chest", f"turn2: must call unlock_chest, got {unlock.name!r}"
    assert unlock.status == OutputStatus.COMPLETED
    # The chaining contract: unlock with the EXACT key find_key returned.
    assert json.loads(unlock.arguments) == {"key": KEY}, (
        f"{model.template_type}: did not chain the returned key into unlock_chest: {unlock.arguments!r}"
    )
    # value-only field_path tagging carries the chained key
    assert turn2["field_args"] == {"key": KEY}, (
        f"{model.template_type}: key field_path tagging wrong: {turn2['field_args']!r}"
    )

    # Turn 3: feed back the chest contents; the model answers, grounded in them.
    conversation += [
        {"type": "function_call", "call_id": unlock.call_id, "name": unlock.name, "arguments": unlock.arguments},
        {"type": "function_call_output", "call_id": unlock.call_id, "output": json.dumps({"contents": CHEST_CONTENTS})},
    ]
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
    assert_or_record(model.template_type, "tool_chaining", "turn3", turn3["events"])
    print_usage_summary([turn1, turn2, turn3])

    assert turn3["order"][0] == "response.created"
    assert turn3["order"][-1] == "done"
    assert turn3["counts"]["response.completed"] == 1

    reasoning_blocks = turn3["added"].get("reasoning", 0)
    if reasoning_blocks:
        assert reasoning_blocks == 1, "turn3: expected at most one reasoning block"
        assert turn3["reasoning"].strip() == turn3["reasoning_done"], "turn3: reasoning deltas != reasoning.done"

    assert turn3["counts"].get("response.function_call_arguments.done", 0) == 0, "turn3: unexpected tool call"
    assert turn3["counts"]["response.output_text.done"] == 1, "turn3: expected one message"
    assert turn3["content"] == turn3["content_done"], "turn3: content deltas != output_text.done"

    msg_open = [item for item in turn3["items_added"] if isinstance(item, OutputMessage)]
    msg_done = [item for item in turn3["items_done"] if isinstance(item, OutputMessage)]
    assert len(msg_open) == 1 and len(msg_done) == 1, "turn3: expected one message item"
    assert msg_open[0].role == "assistant"
    assert not msg_open[0].content, "turn3: message must open with empty content"
    assert msg_open[0].status == OutputStatus.IN_PROGRESS
    assert msg_done[0].status == OutputStatus.COMPLETED

    # The answer must ground in the chest contents, not a hallucination.
    answer = turn3["content_done"].lower()
    assert "jade dragon" in answer or "figurine" in answer, (
        f"{model.template_type}: answer ignored the chest contents: {answer!r}"
    )
