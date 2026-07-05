"""Catch a gpt_oss junk-sample in the act and record its token ids.

Repeats the reason_then_tool turn-1 request (temp 1.0, random seeds,
constrained) with stream_tokens until a run's tool args are not exactly
San Francisco, then dumps the raw generated token ids and their run-text.
Correlate the anomalous ids with the engine log's "corrected token" lines
to split sampler-draw bugs from correction-policy bugs.

Opt-in: GPTOSS_PROBE=1 python -m pytest tests/test_gptoss_anomaly_catch.py -q -s
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


async def test_catch_anomaly(client):
    for attempt in range(12):
        stream = await client.aresponses(
            MODEL_ID,
            stream=True,
            stream_tokens=True,
            input=CONVERSATION,
            core_tools=[GET_WEATHER],
            tool_choice="required",
            max_output_tokens=512,
            reasoning={"effort": "medium"},
            prefix_cache=False,
            deterministic=False,
            temperature=1.0,
            top_p=1.0,
        )
        turn = await drain_stream(stream)
        calls = turn.get("function_calls") or []
        args = calls[0].arguments if calls else "<no call>"
        ids = turn.get("output_token_ids") or []
        chunks = turn.get("output_chunks") or []
        print(f"[catch] attempt {attempt+1}: args={args!r} n_tokens={len(ids)}")
        if '"San Francisco"' not in args:
            print("[catch] ANOMALY — final 25 (id, text):")
            for tid, txt in list(zip(ids, chunks))[-25:]:
                print(f"    id={tid} text={txt!r}")
            return
    print("[catch] no anomaly in 12 attempts")
