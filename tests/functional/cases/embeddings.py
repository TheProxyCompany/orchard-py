import math

import httpx
import pytest

from tests.local_http import local_async_client

pytestmark = pytest.mark.asyncio

REQUEST_TIMEOUT_S = 200.0


def _assert_valid_embedding_response(response_data: dict, model_id: str) -> None:
    assert response_data["object"] == "list"
    assert response_data["model"] == model_id

    assert "data" in response_data
    assert isinstance(response_data["data"], list)
    assert len(response_data["data"]) == 1

    embedding_data = response_data["data"][0]
    assert embedding_data["object"] == "embedding"
    assert embedding_data["index"] == 0

    embedding_vector = embedding_data["embedding"]
    assert isinstance(embedding_vector, list)
    assert embedding_vector, f"{model_id} returned an empty embedding vector"

    for i, value in enumerate(embedding_vector):
        assert isinstance(value, int | float), f"Value at index {i} is not numeric"
        assert math.isfinite(value), f"Value at index {i} is not finite: {value}"

    usage = response_data["usage"]
    assert usage["prompt_tokens"] > 0, f"{model_id} returned zero prompt tokens"
    assert usage["total_tokens"] == usage["prompt_tokens"]


def _extract_error_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text

    if isinstance(body, dict) and "detail" in body:
        return str(body["detail"])
    return str(body)


def _parse_embedding_response(response: httpx.Response, model_id: str) -> dict:
    if response.status_code == 200:
        return response.json()

    detail = _extract_error_detail(response)
    pytest.fail(
        f"Unexpected embeddings response for {model_id}: "
        f"{response.status_code} {detail}"
    )


async def _post_embeddings(
    server_url: str, request_payload: dict, model_id: str
) -> httpx.Response:
    try:
        async with local_async_client(timeout=REQUEST_TIMEOUT_S) as client:
            return await client.post(f"{server_url}/v1/embeddings", json=request_payload)
    except httpx.HTTPError as exc:
        pytest.fail(
            f"Embeddings request failed for {model_id}: {type(exc).__name__}: {exc}"
        )


async def test_embeddings_non_infinite_values(live_server, text_model_id):
    server_url = live_server
    request_payload = {
        "model": text_model_id,
        "input": "Hello, world!",
        "encoding_format": "float",
    }

    response = await _post_embeddings(server_url, request_payload, text_model_id)
    response_data = _parse_embedding_response(response, text_model_id)
    _assert_valid_embedding_response(response_data, text_model_id)


async def test_embeddings_batch_input(live_server, text_model_id):
    server_url = live_server
    request_payload = {
        "model": text_model_id,
        "input": ["Thing 1", "Thing 2"],
        "encoding_format": "float",
    }

    response = await _post_embeddings(server_url, request_payload, text_model_id)
    response_data = _parse_embedding_response(response, text_model_id)
    _assert_valid_embedding_response(response_data, text_model_id)

    # The current endpoint accepts a string list but embeds only the first item.
    assert len(response_data["data"]) == 1


async def test_embeddings_vision_model_text_input(live_server, vision_model_id):
    server_url = live_server
    request_payload = {
        "model": vision_model_id,
        "input": "Hello, world!",
        "encoding_format": "float",
    }

    response = await _post_embeddings(server_url, request_payload, vision_model_id)
    response_data = _parse_embedding_response(response, vision_model_id)
    _assert_valid_embedding_response(response_data, vision_model_id)
