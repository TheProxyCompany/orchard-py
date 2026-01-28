import base64
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.asyncio

MODEL_ID = "moondream3"
APPLE_IMAGE_PATH = Path(__file__).parent / "assets" / "apple.jpg"
MOONDREAM_IMAGE_PATH = Path(__file__).parent / "assets" / "moondream.jpg"


async def test_multimodal_e2e_apple_image(live_server):
    image_bytes = APPLE_IMAGE_PATH.read_bytes()
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    image_data_url = f"data:image/jpeg;base64,{encoded_image}"

    request_payload = {
        "model": MODEL_ID,
        "temperature": 0.0,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What is in this image: "},
                    {"type": "input_image", "image_url": image_data_url},
                ],
            }
        ],
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{live_server}/v1/responses", json=request_payload
        )

    assert response.status_code == 200
    response_data = response.json()

    output_text = "".join(
        content.get("text", "")
        for message in response_data.get("output", [])
        if isinstance(message, dict)
        for content in message.get("content", [])
        if isinstance(content, dict) and content.get("type") == "output_text"
    ).lower()

    print(f"Output text: {output_text}")

    assert "apple" in output_text


async def test_multimodal_e2e_moondream_image(live_server):
    image_bytes = MOONDREAM_IMAGE_PATH.read_bytes()
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    image_data_url = f"data:image/jpeg;base64,{encoded_image}"

    request_payload = {
        "model": MODEL_ID,
        "temperature": 0.0,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": image_data_url},
                    {"type": "input_text", "text": "What is the girl doing?"},
                ],
            }
        ],
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{live_server}/v1/responses", json=request_payload
        )

    assert response.status_code == 200
    response_data = response.json()

    output_text = "".join(
        content.get("text", "")
        for message in response_data.get("output", [])
        if isinstance(message, dict)
        for content in message.get("content", [])
        if isinstance(content, dict) and content.get("type") == "output_text"
    ).lower()

    print(f"Output text: {output_text}")

    assert "burger" in output_text
