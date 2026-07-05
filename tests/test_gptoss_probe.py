"""Cross-check probe: is gpt_oss's "San …" tool arg PIE-specific?

Runs the reason_then_tool turn-1 request against gpt-oss under a settings
sweep and prints each run's tool-call arguments: deterministic (seed 11,
recommended temp 1.0), five non-deterministic runs (random seeds, same
sampling), and greedy. The mlx side runs separately on the identical
rendered prompt.

Opt-in: GPTOSS_PROBE=1 python -m pytest tests/test_gptoss_probe.py -q -s
"""

import os

import pytest

from tests.helpers import drain_stream

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not os.getenv("GPTOSS_PROBE"), reason="opt-in"),
]

MODEL_ID = "openai/gpt-oss-20b"

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

CONVERSATION = [
    {"type": "message", "role": "system", "content": SYSTEM},
    {"type": "message", "role": "user", "content": "What's the weather in San Francisco?"},
]


async def _one(client, tag, **overrides):
    request = dict(
        input=CONVERSATION,
        core_tools=[GET_WEATHER],
        tool_choice="required",
        max_output_tokens=512,
        reasoning={"effort": "medium"},
        prefix_cache=False,
    )
    request.update(overrides)
    stream = await client.aresponses(MODEL_ID, stream=True, stream_tokens=True, **request)
    turn = await drain_stream(stream)
    calls = turn.get("function_calls") or []
    args = calls[0].arguments if calls else "<no tool call>"
    print(f"[probe] {tag}: args={args!r}", flush=True)


async def test_gptoss_arg_probe(client):
    await _one(client, "deterministic seed11 temp1.0", deterministic=True)
    for i in range(5):
        await _one(client, f"random-seed run{i+1} temp1.0", deterministic=False,
                   temperature=1.0, top_p=1.0)
    await _one(client, "greedy temp0", deterministic=False, temperature=0.0)
    for i in range(6):
        await _one(client, f"UNCONSTRAINED run{i+1} temp1.0", deterministic=False,
                   temperature=1.0, top_p=1.0, tool_choice="auto")
