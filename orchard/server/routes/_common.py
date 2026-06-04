import asyncio
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import HTTPException, status
from fastapi.responses import JSONResponse

from orchard.app.ipc_dispatch import QueueRegistration
from orchard.app.model_registry import (
    ModelInfo,
    ModelLoadState,
    ModelRegistry,
    ModelResolutionError,
)
from orchard.ipc.utils import ResponseDeltaDict, release_delta_resources

logger = logging.getLogger(__name__)


async def resolve_model(
    model_registry: ModelRegistry,
    requested_model: str,
) -> tuple[str, ModelInfo]:
    """Resolve a model ID to canonical ID and ModelInfo.

    Raises HTTPException for resolution errors, loading/downloading (202),
    failed (503), and missing runtime info (500).
    """
    try:
        model_state, canonical_id = await model_registry.schedule_model(requested_model)
    except ModelResolutionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": str(exc), "candidates": exc.candidates},
        ) from exc

    if model_state in {ModelLoadState.LOADING, ModelLoadState.ACTIVATING}:
        try:
            model_info = await model_registry.ensure_loaded(canonical_id)
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        return canonical_id, model_info

    if model_state == ModelLoadState.DOWNLOADING:
        raise _ModelNotReadyError(
            JSONResponse(
                status_code=status.HTTP_202_ACCEPTED,
                content={
                    "status": "downloading",
                    "message": "Model download in progress. Retry after a short delay.",
                    "model_id": canonical_id,
                },
                headers={"Retry-After": "30"},
            )
        )

    if model_state == ModelLoadState.FAILED:
        error_detail = model_registry.get_error(canonical_id) or "Model failed to load."
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=error_detail,
        )

    model_info = model_registry.get_if_ready(canonical_id)
    if not model_info:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model '{canonical_id}' reported READY but no runtime info was found.",
        )

    return canonical_id, model_info


class _ModelNotReadyError(Exception):
    """Raised when a model is loading/downloading. Carries a JSONResponse."""

    def __init__(self, response: JSONResponse) -> None:
        self.response = response


@asynccontextmanager
async def managed_stream_session(
    ipc_state: Any,
    request_id: int,
    queue: asyncio.Queue[ResponseDeltaDict],
    *,
    cancel_on_exit: bool = False,
    completed: Callable[[], bool] | None = None,
):
    """Context manager for streaming session lifecycle.

    Registers the queue for the request, and ensures cleanup of any
    leftover deltas (including memoryview release) on exit.
    """
    loop = asyncio.get_running_loop()
    ipc_state.active_request_queues[request_id] = QueueRegistration(
        loop=loop, queue=queue
    )
    try:
        yield queue
    finally:
        if cancel_on_exit and not (completed and completed()):
            try:
                await ipc_state.cancel_request(request_id)
            except Exception:
                logger.debug(
                    "Failed to cancel interrupted PIE request %d.",
                    request_id,
                    exc_info=True,
                )
        ipc_state.active_request_queues.pop(request_id, None)
        try:
            while True:
                leftover = queue.get_nowait()
                release_delta_resources(leftover)
                queue.task_done()
        except asyncio.QueueEmpty:
            pass


def extract_usage(delta: ResponseDeltaDict, counts: dict[str, int]) -> None:
    """Extract token usage from a response delta into counts dict.

    Handles both top-level delta keys and nested usage dicts.
    """

    def _coerce_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    reasoning_tokens = _coerce_int(delta.get("reasoning_tokens"))
    usage_payload = delta.get("usage")
    usage_dict = usage_payload if isinstance(usage_payload, dict) else {}
    if reasoning_tokens is None:
        reasoning_tokens = _coerce_int(usage_dict.get("reasoning_tokens"))
    if reasoning_tokens is not None:
        counts["reasoning_tokens"] = reasoning_tokens

    generation_len = _coerce_int(delta.get("generation_len"))
    if generation_len is None:
        generation_len = _coerce_int(usage_dict.get("generation_len"))
    if generation_len is not None:
        counts["completion_tokens"] = max(
            generation_len - counts.get("reasoning_tokens", 0), 0
        )

    key_map = (
        ("prompt_token_count", "prompt_tokens"),
        ("prompt_tokens", "prompt_tokens"),
        ("input_tokens", "prompt_tokens"),
        ("completion_token_count", "completion_tokens"),
        ("completion_tokens", "completion_tokens"),
        ("output_tokens", "completion_tokens"),
        ("total_token_count", "total_tokens"),
        ("total_tokens", "total_tokens"),
        ("cached_token_count", "cached_tokens"),
    )

    for source_key, target_key in key_map:
        if source_key in delta:
            value = _coerce_int(delta.get(source_key))
            if value is not None:
                counts[target_key] = value

    if isinstance(usage_payload, dict):
        for source_key, target_key in key_map:
            if source_key in usage_payload:
                value = _coerce_int(usage_payload.get(source_key))
                if value is not None:
                    counts[target_key] = value
