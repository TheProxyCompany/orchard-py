import json

import httpx
import pytest

pytestmark = pytest.mark.asyncio

# ---------------------------------------------------------------------------
# 8. Structured output via response format
# ---------------------------------------------------------------------------


async def test_responses_structured_json_schema(live_server, text_model_id):
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
        "model": text_model_id,
        "input": "Return the capital of France and population 2148327 as JSON. Use the integer literal 2148327 without a decimal point.",
        "temperature": 0.0,
        "reasoning": False,
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

    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(f"{live_server}/v1/responses", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"

    message_index = 1 if data["output"][0].get("type") == "reasoning" else 0
    assert len(data["output"]) > message_index
    raw_text = data["output"][message_index]["content"][0]["text"]
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
