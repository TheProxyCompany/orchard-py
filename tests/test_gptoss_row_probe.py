"""Row-integrity probe at sharp positions: sampled token vs own top-k.

Chat route with the get_weather tool at temp 1.0 (random seeds): for every
step, flag sampled ids missing from the engine's own reported top-10 AT
CONFIDENT STEPS (p_top1 >= 0.9). Legit tail draws only escape top-10 at
flat steps; a confident-step escape means the sampled id and the logged
row disagree.

Opt-in: GPTOSS_PROBE=1 python -m pytest tests/test_gptoss_row_probe.py -q -s
"""

import math
import os

import pytest

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

MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant with tool calling. "
     "Reason about the request, then call a tool when needed and use its result to answer."},
    {"role": "user", "content": "What's the weather in San Francisco?"},
]


async def test_sharp_row_integrity(engine):
    client = engine.client(MODEL_ID)
    total = 0
    confident_misses = []
    junk_runs = 0
    for run in range(10):
        stream = client.chat(
            MODEL_ID,
            MESSAGES,
            stream=True,
            temperature=1.0,
            top_p=1.0,
            max_generated_tokens=200,
            top_logprobs=10,
            core_tools=[GET_WEATHER],
            tool_choice="required",
        )
        step = 0
        text = []
        trace = []
        for delta in stream:
            if delta.content:
                text.append(delta.content)
            if not delta.tokens:
                continue
            step += 1
            total += 1
            top = []
            if delta.top_logprobs:
                top = [(int(float(e["token"])), float(e["probability"])) for e in delta.top_logprobs]
            trace.append((delta.tokens[0], top[:3]))
            ids = [t for t, _ in top]
            sampled = delta.tokens[0]
            p1 = math.exp(top[0][1]) if top else 0.0
            if top and sampled not in ids and p1 >= 0.9:
                confident_misses.append((run, step, sampled, top[:3], round(p1, 4)))
        joined = "".join(text)
        bad = (" " in joined) or (" " in joined) or ("…" in joined)
        junk_runs += bad
        print(f"[sharp] run{run}: junk={'YES' if bad else 'no'} steps={step}")
        if bad:
            import json as _json
            with open(f"/tmp/junk_trace_run{run}.json", "w") as f:
                _json.dump({"token_ids": [t for t, _ in trace]}, f)
            print(f"[sharp] run{run} trace (id, top3):")
            for t, tops in trace:
                print(f"    id={t} top3={tops}")
    client.close()
    print(f"\n[sharp] total steps={total} confident-misses={len(confident_misses)} junk_runs={junk_runs}/10")
    for run, step, sampled, top3, p1 in confident_misses[:12]:
        print(f"  run{run} step{step}: sampled={sampled} row_top3={top3} p_top1={p1}")
