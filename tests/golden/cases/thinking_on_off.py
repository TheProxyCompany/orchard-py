import pytest
from tests.golden.golden_io import assert_or_record
from tests.helpers import drain_stream, print_usage_summary, render_prompt_blue
from tests.models import Model

from orchard.clients.client import Client
from orchard.server.models.responses import (
    OutputMessage,
    OutputReasoning,
    OutputStatus,
    ResponseCompletedEvent,
)

pytestmark = pytest.mark.asyncio

SYSTEM = "You are a careful assistant. Answer the user's question correctly."
QUESTION = "What is 17 + 26? Reply with just the number."
ANSWER = "43"


def _reasoning_tokens(turn: dict) -> int:
    """Pull usage.output_tokens_details.reasoning_tokens off the completed event.

    drain_stream only surfaces input/output/cached; the thinking-token count
    lives in the response usage detail, which is what the on/off split is about.
    """
    completed = [e for e in turn["events"] if isinstance(e, ResponseCompletedEvent)]
    assert len(completed) == 1, "expected exactly one response.completed"
    usage = completed[0].response.usage
    assert usage is not None, "response.completed carried no usage"
    details = usage.output_tokens_details
    return details.reasoning_tokens if details is not None else 0


async def test_thinking_on_off(client: Client, model: Model):
    """Same prompt, thinking ON then OFF — pins that the off switch silences it.

    ON (reasoning={"effort":"medium"}): exactly one reasoning block, deltas
    accumulate, usage reasoning_tokens > 0. OFF (reasoning=False): zero reasoning
    items, zero reasoning deltas, usage reasoning_tokens == 0. Both still answer
    correctly. Deterministic, streamed, recorded as turns "on" / "off".
    """
    if not model.thinking:
        return
    if model.thinking == "required":
        return
    print(
        f"\n\033[1;33m━━━ {model.template_type} · thinking on → off ━━━\033[0m",
        flush=True,
    )

    conversation = [
        {"type": "message", "role": "system", "content": SYSTEM},
        {"type": "message", "role": "user", "content": QUESTION},
    ]

    # Turn 1 ("on"): thinking enabled — the model reasons, then answers.
    on_request = dict(
        input=conversation,
        deterministic=True,
        max_output_tokens=512,
        reasoning={"effort": "medium"},
        prefix_cache=False,
    )
    await render_prompt_blue(client, model.checkpoint, **on_request)
    stream = await client.aresponses(
        model.checkpoint, stream=True, stream_tokens=True, **on_request
    )
    on = await drain_stream(stream)
    assert_or_record(model.template_type, "thinking_on_off", "on", on["events"])

    assert on["order"][0] == "response.created"
    assert on["order"][-1] == "done"
    assert on["counts"]["response.created"] == 1
    assert on["counts"]["response.in_progress"] == 1
    assert on["counts"]["response.completed"] == 1

    assert on["added"]["reasoning"] == 1, "on: expected exactly one reasoning block"
    assert on["counts"]["response.reasoning.done"] == 1
    assert on["counts"]["response.reasoning.delta"] >= 1, "on: reasoning produced no deltas"
    assert on["reasoning"].strip() == on["reasoning_done"], "on: reasoning deltas != reasoning.done"
    assert "<|" not in on["reasoning"] and "</" not in on["reasoning"], "on: control leak in reasoning"

    on_reasoning = [item for item in on["items_added"] if isinstance(item, OutputReasoning)]
    assert len(on_reasoning) == 1, "on: expected one reasoning item opened"
    assert on_reasoning[0].status == OutputStatus.IN_PROGRESS
    assert _reasoning_tokens(on) > 0, "on: usage reported zero reasoning tokens while thinking"

    assert on["counts"]["response.output_text.done"] == 1, "on: expected one message"
    assert on["content"] == on["content_done"], "on: content deltas != output_text.done"
    on_msg = [item for item in on["items_done"] if isinstance(item, OutputMessage)]
    assert len(on_msg) == 1 and on_msg[0].status == OutputStatus.COMPLETED
    assert ANSWER in on["content_done"], f"on: wrong answer: {on['content_done']!r}"

    # Turn 2 ("off"): thinking suppressed (reasoning=False) — same prompt, no CoT.
    off_request = dict(
        input=conversation,
        deterministic=True,
        max_output_tokens=512,
        reasoning=False,
        prefix_cache=False,
    )
    await render_prompt_blue(client, model.checkpoint, **off_request)
    stream = await client.aresponses(
        model.checkpoint, stream=True, stream_tokens=True, **off_request
    )
    off = await drain_stream(stream)
    print_usage_summary([on, off])
    assert_or_record(model.template_type, "thinking_on_off", "off", off["events"])

    assert off["order"][0] == "response.created"
    assert off["order"][-1] == "done"
    assert off["counts"]["response.created"] == 1
    assert off["counts"]["response.in_progress"] == 1
    assert off["counts"]["response.completed"] == 1

    assert off["added"].get("reasoning", 0) == 0, "off: thinking still produced a reasoning block"
    assert off["counts"].get("response.reasoning.delta", 0) == 0, "off: reasoning deltas leaked"
    assert off["counts"].get("response.reasoning.done", 0) == 0, "off: reasoning.done leaked"
    assert not [item for item in off["items_added"] if isinstance(item, OutputReasoning)], "off: a reasoning item was opened"
    assert _reasoning_tokens(off) == 0, "off: usage charged reasoning tokens with thinking off"

    assert off["counts"]["response.output_text.done"] == 1, "off: expected one message"
    assert off["content"] == off["content_done"], "off: content deltas != output_text.done"
    off_msg = [item for item in off["items_done"] if isinstance(item, OutputMessage)]
    assert len(off_msg) == 1 and off_msg[0].status == OutputStatus.COMPLETED
    assert ANSWER in off["content_done"], f"off: wrong answer: {off['content_done']!r}"
