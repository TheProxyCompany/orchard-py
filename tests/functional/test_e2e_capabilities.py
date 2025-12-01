from pathlib import Path

import pytest
from PIL import Image

from proxy_inference_engine.clients.moondream import MoondreamClient
from proxy_inference_engine.engine.inference_engine import InferenceEngine

APPLE_IMAGE_PATH = Path(__file__).parent / "assets" / "apple.jpg"
MOONDREAM_IMAGE_PATH = Path(__file__).parent / "assets" / "moondream.jpg"
BOTTLES_IMAGE_PATH = Path(__file__).parent / "assets" / "bottles.jpg"


def test_moondream_reasoning_grounding(engine: InferenceEngine):
    client = engine.client(MoondreamClient.model_id)
    assert isinstance(client, MoondreamClient)
    try:
        with Image.open(BOTTLES_IMAGE_PATH) as img:
            image = img.convert("RGB")

        response = client.query(
            image=image,
            prompt="How many bottles are shown?",
            reasoning=True,
            temperature=0.0,
        )
    finally:
        client.close()

    assert "answer" in response
    assert "grounding" in response["reasoning"]
    assert "text" in response["reasoning"]

    answer_text = str(response["answer"]).lower()
    reasoning_text = str(response["reasoning"]["text"]).lower()
    grounding = response["reasoning"]["grounding"]

    print(grounding)
    print(reasoning_text)
    print(answer_text)

    assert "6" in answer_text, f"Answer text should contain '6', but got {answer_text}"
    assert isinstance(grounding, list)
    assert grounding, "Grounding output should not be empty"

    for ground in grounding:
        assert isinstance(ground, dict)
        assert "points" in ground
        assert "start_idx" in ground
        assert "end_idx" in ground
        assert len(ground.get("points", [])) == 6, (
            f"Expected 6 points, got {len(ground.get('points', []))}"
        )

    assert "duckhorn" in reasoning_text, (
        f"Model should mention the label on the bottle, but got:\n{reasoning_text}"
    )


@pytest.mark.parametrize("length", ["normal", "short", "long"])
def test_moondream_caption(engine: InferenceEngine, length: str):
    """Test caption generation for an image."""
    client = engine.client(MoondreamClient.model_id)
    assert isinstance(client, MoondreamClient)
    try:
        with Image.open(APPLE_IMAGE_PATH) as img:
            image = img.convert("RGB")

        result = client.caption(
            image,
            length=length,
            temperature=0.0,
        )
    finally:
        client.close()

    assert "caption" in result
    assert isinstance(result["caption"], str)
    assert len(result["caption"]) > 0
    caption = result["caption"].lower()
    assert "apple" in caption
    print(f"{length.capitalize()} caption: {caption}")


def test_moondream_detect(engine: InferenceEngine):
    """Test object detection with bounding boxes."""
    client = engine.client(MoondreamClient.model_id)
    assert isinstance(client, MoondreamClient)
    try:
        with Image.open(APPLE_IMAGE_PATH) as img:
            image = img.convert("RGB")

        result = client.detect(image, object="apple", temperature=0.0)
    finally:
        client.close()

    assert "objects" in result
    assert isinstance(result["objects"], list)
    objects = result["objects"]
    assert len(objects) > 0
    print(objects)


def test_moondream_query_with_spatial_refs(engine: InferenceEngine):
    """Test query with spatial reference (point)."""
    client = engine.client(MoondreamClient.model_id)
    assert isinstance(client, MoondreamClient)
    try:
        with Image.open(APPLE_IMAGE_PATH) as img:
            image = img.convert("RGB")

        result = client.query(
            image,
            prompt="What is at this location?",
            spatial_refs=[(0.5, 0.5)],
            temperature=0.0,
        )
    finally:
        client.close()

    assert "answer" in result
    answer = result["answer"].lower()
    print(answer)
    assert "apple" in answer


def test_moondream_point(engine: InferenceEngine):
    """Test pointing to objects in an image."""
    client = engine.client(MoondreamClient.model_id)
    assert isinstance(client, MoondreamClient)
    try:
        with Image.open(MOONDREAM_IMAGE_PATH) as img:
            image = img.convert("RGB")

        result = client.point(image, object="Eyes", temperature=0.0)
    finally:
        client.close()

    assert "points" in result
    assert isinstance(result["points"], list)
    points = result["points"]
    assert len(points) > 0
    print(points)


def test_moondream_detect_gaze(engine: InferenceEngine):
    """Test gaze detection with deterministic output (temp=0)."""
    client = engine.client(MoondreamClient.model_id)
    assert isinstance(client, MoondreamClient)
    try:
        with Image.open(MOONDREAM_IMAGE_PATH) as img:
            image = img.convert("RGB")

        # Eye position - center of the character's face area
        eye_position = (0.5419921875, 0.5419921875)

        result = client.detect_gaze(
            image,
            eye=eye_position,
            temperature=0.0,
            max_completion_tokens=100,
        )
    finally:
        client.close()

    assert "gaze" in result
    gaze = result["gaze"]
    assert isinstance(gaze, dict)
    assert "x" in gaze and "y" in gaze
    # baseline implementation of gaze detection returns the center of the image
    assert gaze["x"] == 0.5
    assert gaze["y"] == 0.5
    print(f"Gaze: ({gaze['x']}, {gaze['y']})")
