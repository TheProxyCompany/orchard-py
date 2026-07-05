"""Position probe: PIE's next-token distribution after '{"location":"San'.

Feeds the byte-identical prefix mlx used (/tmp/gptoss_prefix.txt) through
/v1/completions with apply_chat_template=False and samples one token 24
times at temp 1.0, then once greedy with logprobs.

Opt-in: GPTOSS_PROBE=1 python -m pytest tests/test_gptoss_position_probe.py -q -s
"""

import collections
import os

import httpx
import pytest

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not os.getenv("GPTOSS_PROBE"), reason="opt-in"),
]

MODEL_ID = "openai/gpt-oss-20b"


async def test_position_distribution(live_server):
    prefix = open("/tmp/gptoss_prefix.txt").read()
    counts = collections.Counter()
    async with httpx.AsyncClient(timeout=180.0) as client:
        for _ in range(24):
            r = await client.post(
                f"{live_server}/v1/completions",
                json={
                    "model": MODEL_ID,
                    "prompt": prefix,
                    "apply_chat_template": False,
                    "max_completion_tokens": 1,
                    "temperature": 1.0,
                    "top_p": 1.0,
                },
            )
            assert r.status_code == 200, r.text
            counts[r.json()["choices"][0]["text"]] += 1
        greedy = await client.post(
            f"{live_server}/v1/completions",
            json={
                "model": MODEL_ID,
                "prompt": prefix,
                "apply_chat_template": False,
                "max_completion_tokens": 1,
                "temperature": 0.0,
                "logprobs": 15,
            },
        )
    print("\n[position] sample counts at temp 1.0 (n=24):")
    for tok, n in counts.most_common():
        print(f"  {n:3d}x {tok!r}")
    body = greedy.json()["choices"][0]
    print(f"[position] greedy text={body['text']!r} logprobs={body.get('logprobs')}")
