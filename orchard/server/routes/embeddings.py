import asyncio
import logging
import random
import struct
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from orchard.app.ipc_dispatch import QueueRegistration
from orchard.ipc.serialization import _build_request_payload
from orchard.ipc.utils import (
    ResponseDeltaDict,
    release_delta_resources,
)
from orchard.server.dependencies import IPCStateDep, ModelRegistryDep
from orchard.server.exceptions import InferenceError
from orchard.server.models.embeddings import (
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
)
from orchard.server.routes._common import _ModelNotReadyError, resolve_model

logger = logging.getLogger(__name__)

embeddings_router = APIRouter()


@embeddings_router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(
    request: EmbeddingRequest,
    ipc_state: IPCStateDep,
    model_registry: ModelRegistryDep,
) -> EmbeddingResponse | JSONResponse:
    """
    Handles requests to the `/v1/embeddings` endpoint.
    """
    logger.info("Embeddings request for model: %s", request.model)

    # For now, we'll handle single string input only as a placeholder
    # Full implementation will handle batching and tokenization
    if isinstance(request.input, str):
        input_text = request.input
    elif (
        isinstance(request.input, list)
        and len(request.input) > 0
        and isinstance(request.input[0], str)
    ):
        # For MVP, just take the first string
        input_text = request.input[0]
        logger.warning("Batch embedding not yet implemented, using first input only")
    else:
        raise HTTPException(
            status_code=400,
            detail="Only string input is currently supported for embeddings",
        )

    # 1. Generate request ID
    current_request_id = await ipc_state.get_next_request_id()
    logger.debug("Generated request ID: %d", current_request_id)

    try:
        canonical_id, model_info = await resolve_model(model_registry, request.model)
    except _ModelNotReadyError as exc:
        return exc.response

    response_channel_id = ipc_state.response_channel_id or current_request_id
    rng_seed = random.randint(0, 2**32 - 1)
    prompt_bytes = input_text.encode("utf-8")
    layout_segments = [
        {
            "type": "text",
            "length": len(prompt_bytes),
        }
    ]

    prompt_payload = {
        "prompt_bytes": prompt_bytes,
        "image_buffers": [],
        "layout": layout_segments,
        "sampling_params": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "rng_seed": rng_seed,
        },
        "logits_params": {
            "top_logprobs": 0,
            "frequency_penalty": 0.0,
            "logit_bias": {},
            "presence_penalty": 0.0,
            "repetition_context_size": 0,
            "repetition_penalty": 1.0,
        },
        "max_generated_tokens": 0,
        "stop_sequences": [],
        "tool_schemas_json": "",
        "response_format_json": "",
        "num_candidates": 1,
        "best_of": 1,
        "final_candidates": 1,
    }

    request_bytes = _build_request_payload(
        request_id=current_request_id,
        model_id=canonical_id,
        model_path=model_info.model_path,
        request_type="embedding",
        response_channel_id=response_channel_id,
        prompts=[prompt_payload],
    )

    # 3. Create and register the asyncio Queue for this request
    response_queue = asyncio.Queue[ResponseDeltaDict]()
    loop = asyncio.get_running_loop()
    ipc_state.active_request_queues[current_request_id] = QueueRegistration(
        loop=loop, queue=response_queue
    )
    logger.debug("Registered queue for request ID: %d", current_request_id)

    # 4. Submit request to C++ engine
    try:
        logger.debug(
            "Submitting embedding request %d to C++ engine", current_request_id
        )
        socket = ipc_state.request_socket
        if socket is None:
            raise RuntimeError("Request socket is not initialized.")

        await socket.asend(request_bytes)
        logger.info("Submitted embedding request %d successfully", current_request_id)

        # 5. Gather response (embeddings are always non-streaming)
        response_data = await gather_embedding_response(
            current_request_id, response_queue
        )

        usage = EmbeddingUsage(
            prompt_tokens=response_data["prompt_tokens"],
            total_tokens=response_data["prompt_tokens"],
            # No completion tokens for embeddings
        )

        embedding_vector = response_data.get("embedding_vector", [])

        embedding_data = EmbeddingData(embedding=embedding_vector, index=0)

        final_response = EmbeddingResponse(
            object="list",
            data=[embedding_data],
            model=request.model,
            usage=usage,
        )

        logger.info("Embedding request %d completed successfully.", current_request_id)
        return final_response

    except Exception as e:
        logger.error("Error processing embedding request: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during embedding generation.",
        ) from e
    finally:
        # Clean up the request queue
        if current_request_id in ipc_state.active_request_queues:
            _ = ipc_state.active_request_queues.pop(current_request_id, None)
            logger.debug(
                "Cleaned up queue for embedding request ID %d.", current_request_id
            )


async def gather_embedding_response(
    request_id: int,
    queue: asyncio.Queue[ResponseDeltaDict],
) -> dict[str, Any]:
    """Collects the embedding response from the queue."""
    prompt_tokens = 0
    embedding_vector = []

    while True:
        try:
            delta = await asyncio.wait_for(queue.get(), timeout=30.0)
        except TimeoutError as e:
            logger.error(
                "Timeout waiting for embedding response for request %d", request_id
            )
            raise InferenceError(
                "Timeout receiving embedding response from engine."
            ) from e
        else:
            try:
                logger.debug("Embedding response received for request %d", request_id)

                if delta.get("prompt_token_count", 0) > 0:
                    prompt_tokens = delta["prompt_token_count"]

                if delta.get("embedding_bytes"):
                    embedding_bytes = delta["embedding_bytes"]
                    num_floats = len(embedding_bytes) // 4
                    embedding_vector = list(
                        struct.unpack(f"{num_floats}f", embedding_bytes)
                    )
                    logger.debug("Unpacked %d floats from embedding bytes.", num_floats)

                if delta.get("is_final_delta", False):
                    logger.debug(
                        "Received final delta for embedding request %d", request_id
                    )
                    break
            finally:
                queue.task_done()
                release_delta_resources(delta)

    return {
        "prompt_tokens": prompt_tokens,
        "embedding_vector": embedding_vector,
    }
