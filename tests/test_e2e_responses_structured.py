import json

import httpx
import pytest

pytestmark = pytest.mark.asyncio

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


# ---------------------------------------------------------------------------
# 8. Structured output via response format
# ---------------------------------------------------------------------------


async def test_responses_structured_json_schema(live_server):
    """The text field with json_schema format produces valid structured output."""
    schema = {
        "type": "object",
        "properties": {
            "capital": {"type": "string"},
            "population": {"type": "integer"},
        },
        "required": ["capital", "population"],
    }

    payload = {
        "model": MODEL_ID,
        "input": "What is the capital of France and its approximate population? Respond as JSON.",
        "temperature": 0.0,
        "max_output_tokens": 64,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "city_info",
                "schema": schema,
                "strict": True,
            }
        },
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(f"{live_server}/v1/responses", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"

    raw_text = data["output"][0]["content"][0]["text"]
    # Find JSON in the output
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    assert start != -1 and end != -1, f"No JSON found in output: {raw_text}"

    parsed = json.loads(raw_text[start : end + 1])
    assert isinstance(parsed, dict)
    assert "capital" in parsed
    assert "population" in parsed
    assert isinstance(parsed["capital"], str)
    assert isinstance(parsed["population"], int)
    assert "paris" in parsed["capital"].lower()
    print(f"Structured output: {parsed}")
