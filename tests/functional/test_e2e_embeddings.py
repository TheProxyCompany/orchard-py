# import pytest
# import httpx
# import math

# pytestmark = pytest.mark.asyncio


# async def test_embeddings_non_infinite_values(live_server):
#     """
#     Tests the embeddings endpoint returns valid, non-infinite floating point values.
#     Verifies the embedding vector contains only finite numbers.
#     """
#     server_url = live_server
#     request_payload = {
#         "model": "test-model",
#         "input": "Hello, world!",
#         "encoding_format": "float",
#     }

#     async with httpx.AsyncClient(timeout=30.0) as client:
#         response = await client.post(
#             f"{server_url}/v1/embeddings", json=request_payload
#         )

#     # --- Assertions ---
#     assert response.status_code == 200
#     response_data = response.json()

#     # Verify response structure
#     assert "object" in response_data
#     assert response_data["object"] == "list"
#     assert "data" in response_data
#     assert isinstance(response_data["data"], list)
#     assert len(response_data["data"]) > 0

#     # Check first embedding
#     embedding_data = response_data["data"][0]
#     assert "object" in embedding_data
#     assert embedding_data["object"] == "embedding"
#     assert "embedding" in embedding_data
#     assert "index" in embedding_data
#     assert embedding_data["index"] == 0

#     # Verify embedding vector
#     embedding_vector = embedding_data["embedding"]
#     assert isinstance(embedding_vector, list)
#     assert len(embedding_vector) > 0

#     # Check all values are finite (not inf, -inf, or nan)
#     for i, value in enumerate(embedding_vector):
#         assert isinstance(value, int | float), f"Value at index {i} is not a number"
#         assert math.isfinite(value), f"Value at index {i} is not finite: {value}"

#     # Verify model field
#     assert "model" in response_data
#     assert response_data["model"] == request_payload["model"]

#     # Verify usage statistics
#     assert "usage" in response_data
#     usage = response_data["usage"]
#     assert "prompt_tokens" in usage
#     assert "total_tokens" in usage
#     assert usage["prompt_tokens"] > 0
#     assert (
#         usage["total_tokens"] == usage["prompt_tokens"]
#     )  # No completion tokens for embeddings


# async def test_embeddings_batch_input(live_server):
#     """
#     Tests the embeddings endpoint with a batch input.
#     """
#     server_url = live_server
#     request_payload = {
#         "model": "test-model",
#         "input": ["Thing 1", "Thing 2"],
#         "encoding_format": "float"
#     }

#     async with httpx.AsyncClient(timeout=30.0) as client:
#         response = await client.post(
#             f"{server_url}/v1/embeddings", json=request_payload
#         )

#     assert response.status_code == 200
