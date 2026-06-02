import json

import pytest
from golden_io import assert_or_record
from helpers import drain_stream, print_usage_summary, render_prompt_blue
from models import Model

from orchard.clients.client import Client
from orchard.server.models.responses import (
    OutputMessage,
    OutputStatus,
)

pytestmark = pytest.mark.asyncio

# A small, fixed structured object whose value is fully determined by the prompt.
# The schema is strict so the engine's PSE-constrained decode must emit exactly
# these two fields, integer-typed, and nothing else.
CITY_SCHEMA = {
    "type": "object",
    "properties": {
        "capital": {"type": "string"},
        "population": {"type": "integer"},
    },
    "required": ["capital", "population"],
    "additionalProperties": False,
}

EXPECTED = {"capital": "Paris", "population": 2148327}

SYSTEM = (
    "You are a helpful assistant. Reason about the request, then return the "
    "answer as a single JSON object that matches the requested schema exactly."
)

USER = (
    "Return the capital of France and its population. Use the capital string "
    '"Paris" and the integer literal 2148327 (no decimal point).'
)


async def test_reason_then_structured(client: Client, model: Model):
    """reason -> strict json_schema structured object, streamed.

    Pins the streaming event lifecycle (exactly-once reasoning block that
    terminates cleanly, delta accumulation, no control-token leak) and the
    semantic outcome (the emitted object parses to the EXACT expected value).
    Targets the Gemma-harmony / PSE non-termination class — the structured
    equality is exact, so a failing golden means the engine is wrong.
    """
    if not model.thinking:
        return
    reasoning = {"effort": "medium"}
    print(
        f"\n\033[1;33m━━━ {model.template_type} · reason → structured · single-turn ━━━\033[0m",
        flush=True,
    )

    conversation = [
        {"type": "message", "role": "system", "content": SYSTEM},
        {"type": "message", "role": "user", "content": USER},
    ]

    turn1_request = dict(
        input=conversation,
        text={
            "format": {
                "type": "json_schema",
                "name": "city_info",
                "schema": CITY_SCHEMA,
                "strict": True,
            }
        },
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
    print_usage_summary([turn1])
    assert_or_record(model.template_type, "reason_then_structured", "turn1", turn1["events"])

    assert turn1["order"][0] == "response.created"
    assert turn1["order"][-1] == "done"
    assert turn1["counts"]["response.created"] == 1
    assert turn1["counts"]["response.in_progress"] == 1
    assert turn1["counts"]["response.completed"] == 1

    # Exactly one reasoning block that terminates cleanly: deltas accumulate to
    # the .done text, and no control/template token leaks into the reasoning.
    assert turn1["added"]["reasoning"] == 1, "turn1: expected exactly one reasoning block"
    assert turn1["counts"]["response.reasoning.done"] == 1, "turn1: reasoning did not terminate cleanly"
    assert turn1["counts"]["response.reasoning.delta"] >= 1
    assert turn1["reasoning"].strip() == turn1["reasoning_done"], "turn1: reasoning deltas != reasoning.done"
    assert "<|" not in turn1["reasoning"] and "</" not in turn1["reasoning"], "turn1: control leak in reasoning"

    # The structured answer comes back as one assistant message, no tool call.
    assert turn1["counts"].get("response.function_call_arguments.done", 0) == 0, "turn1: unexpected tool call"
    assert turn1["counts"]["response.output_text.done"] == 1, "turn1: expected one message"
    assert turn1["content"] == turn1["content_done"], "turn1: content deltas != output_text.done"

    # message lifecycle the UI streams: opens empty, fills via deltas, closes completed.
    msg_open = [item for item in turn1["items_added"] if isinstance(item, OutputMessage)]
    msg_done = [item for item in turn1["items_done"] if isinstance(item, OutputMessage)]
    assert len(msg_open) == 1 and len(msg_done) == 1, "turn1: expected one message item"
    assert msg_open[0].role == "assistant"
    assert not msg_open[0].content, "turn1: message must open with empty content"
    assert msg_open[0].status == OutputStatus.IN_PROGRESS
    assert msg_done[0].status == OutputStatus.COMPLETED

    # Strict structured output: the emitted text parses to EXACTLY the expected object.
    assert json.loads(turn1["content_done"]) == EXPECTED, (
        f"{model.template_type}: structured output != expected: {turn1['content_done']!r}"
    )
