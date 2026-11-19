import asyncio
import base64
import logging
import random
import struct
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from orchard.app.ipc_dispatch import EngineDisconnectedError
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
from orchard.server.routes._common import _ModelNotReadyError, managed_stream_session, resolve_model

logger = logging.getLogger(__name__)

embeddings_router = APIRouter()


def _coerce_embedding_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, str):
        return base64.b64decode(value)
    if isinstance(value, list):
        return bytes(value)
    raise InferenceError("Embedding response contained an unsupported byte payload.")


def _embedding_prompt(text: str, rng_seed: int) -> dict[str, Any]:
    prompt_bytes = text.encode("utf-8")
    layout_segments = [
        {
            "type": "text",
            "length": len(prompt_bytes),
        }
    ]

    return {
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

    if isinstance(request.input, str):
        texts = [request.input]
    elif isinstance(request.input, list) and request.input and all(isinstance(text, str) for text in request.input):
        texts = request.input
    else:
        raise HTTPException(status_code=400, detail="Expected a string or a nonempty list of strings.")

    # 1. Generate request ID
    current_request_id = await ipc_state.get_next_request_id()
    logger.debug("Generated request ID: %d", current_request_id)

    try:
        canonical_id, model_info = await resolve_model(model_registry, request.model)
    except _ModelNotReadyError as exc:
        return exc.response

    response_channel_id = ipc_state.response_channel_id or current_request_id
    rng_seed = random.randint(0, 2**32 - 1)
    request_bytes = _build_request_payload(
        request_id=current_request_id,
        model_id=canonical_id,
        model_path=model_info.model_path,
        request_type="embedding",
        response_channel_id=response_channel_id,
        lossless_responses=ipc_state.lossless_responses,
        prompts=[_embedding_prompt(text, rng_seed) for text in texts],
    )

    response_queue = asyncio.Queue[ResponseDeltaDict]()
    completed = False
    try:
        async with managed_stream_session(
            ipc_state, current_request_id, response_queue,
            cancel_on_exit=True, completed=lambda: completed,
        ):
            logger.debug("Submitting embedding request %d to C++ engine", current_request_id)
            await ipc_state.send_request(request_bytes)
            response_data = await gather_embedding_response(
                current_request_id, response_queue, ipc_state, len(texts)
            )
            completed = True

        usage = EmbeddingUsage(
            prompt_tokens=response_data["prompt_tokens"],
            total_tokens=response_data["prompt_tokens"],
            # No completion tokens for embeddings
        )

        final_response = EmbeddingResponse(
            object="list",
            data=[
                EmbeddingData(embedding=vector, index=index)
                for index, vector in enumerate(response_data["embedding_vectors"])
            ],
            model=request.model,
            usage=usage,
        )

        logger.info("Embedding request %d completed successfully.", current_request_id)
        return final_response
    except (InferenceError, EngineDisconnectedError) as exc:
        logger.error(
            "Inference error processing embedding request %d: %s",
            current_request_id,
            exc,
        )
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as e:
        logger.error("Error processing embedding request: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during embedding generation.",
        ) from e


async def gather_embedding_response(
    request_id: int,
    queue: asyncio.Queue[ResponseDeltaDict],
    ipc_state: IPCStateDep,
    prompt_count: int = 1,
) -> dict[str, Any]:
    """Collect one final vector per prompt, independent of response arrival order."""
    vectors: list[list[float] | None] = [None] * prompt_count
    tokens = [0] * prompt_count
    finished: set[int] = set()

    while len(finished) < prompt_count:
        try:
            delta = await ipc_state.next_delta(queue)
        except EngineDisconnectedError as exc:
            raise InferenceError(str(exc)) from exc
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

                if delta.get("error"):
                    raise InferenceError(str(delta["error"]))

                status_value = str(delta.get("status", "")).lower()
                finish_value = str(delta.get("finish_reason", "")).lower()
                if status_value == "error" or finish_value == "error":
                    raise InferenceError(
                        str(
                            delta.get("content")
                            or delta.get("message")
                            or delta.get("error")
                            or "Embedding generation failed."
                        )
                    )

                index = delta.get("prompt_index")
                if index is None and prompt_count == 1:
                    index = 0
                if not isinstance(index, int) or not 0 <= index < prompt_count:
                    raise InferenceError(f"Invalid embedding prompt index {index!r} for {prompt_count} inputs.")
                if index in finished:
                    continue
                if delta.get("prompt_token_count", 0) > 0:
                    tokens[index] = delta["prompt_token_count"]

                if delta.get("embedding_bytes"):
                    raw = _coerce_embedding_bytes(delta["embedding_bytes"])
                    if len(raw) % 4:
                        raise InferenceError("Embedding payload contains an incomplete float32 value.")
                    vectors[index] = list(struct.unpack(f"{len(raw) // 4}f", raw))

                if delta.get("is_final_delta", False):
                    if vectors[index] is None:
                        raise InferenceError(f"Missing embedding vector for prompt index {index}.")
                    finished.add(index)
            finally:
                queue.task_done()
                release_delta_resources(delta)

    return {
        "prompt_tokens": sum(tokens),
        "embedding_vectors": vectors,
        "embedding_vector": vectors[0] if vectors else [],
    }
