import json

import httpx
import pytest

pytestmark = pytest.mark.asyncio

MODEL_ID = "moondream3"


@pytest.mark.parametrize("temperature", [0.0])
async def test_chat_completion_structured_json_response(live_server, temperature):
    server_url = live_server
    schema = {
        "type": "object",
        "properties": {
            "color": {
                "type": "object",
                "properties": {
                    "R": {"type": "integer", "minimum": 0, "maximum": 255},
                    "G": {"type": "integer", "minimum": 0, "maximum": 255},
                    "B": {"type": "integer", "minimum": 0, "maximum": 255},
                },
                "required": ["R", "G", "B"],
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["color", "confidence"],
    }
    payload = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": f"Respond with a JSON object with a rgb(r, g, b) color and a confidence score. use this schema: {json.dumps(schema, indent=2)}. Make it pretty, this should reflect your internal associations with the color.",
            }
        ],
        "temperature": temperature,
        "stream": False,
        "logprobs": True,
        "top_logprobs": 3,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "color_summary",
                "strict": True,
                "schema": schema,
            },
        },
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{server_url}/v1/chat/completions",
            json=payload,
        )

    assert response.status_code == 200
    body = response.json()
    assert body.get("choices")

    content = body["choices"][0]["message"]["content"]
    assert content, "Expected structured content in completion."

    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Invalid JSON in completion: {content}")

    json_payload = content[start : end + 1]
    parsed = json.loads(json_payload)
    assert isinstance(parsed, dict)
    assert "color" in parsed
    assert isinstance(parsed["color"], dict)
    assert "R" in parsed["color"]
    assert isinstance(parsed["color"]["R"], int | float)
    assert "G" in parsed["color"]
    assert isinstance(parsed["color"]["G"], int | float)
    assert "B" in parsed["color"]
    assert isinstance(parsed["color"]["B"], int | float)
    if "confidence" in parsed:
        assert isinstance(parsed["confidence"], int | float)
    r = int(parsed["color"]["R"])
    g = int(parsed["color"]["G"])
    b = int(parsed["color"]["B"])
    if "confidence" in parsed:
        confidence = parsed["confidence"]
        opacity = int(confidence * 255)
        colored_text = f"\033[38;2;{r};{g};{b};{opacity}m{MODEL_ID}'s color is rgb({r}, {g}, {b}) with confidence {confidence} at temperature {temperature}.\033[0m"
    else:
        colored_text = f"\033[38;2;{r};{g};{b}m{MODEL_ID}'s color is rgb({r}, {g}, {b}) at temperature {temperature}. No confidence score provided.\033[0m"
    print(colored_text)
