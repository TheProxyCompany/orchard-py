import logging

from fastapi import APIRouter
from pydantic import BaseModel

from proxy_inference_engine.server.dependencies import ModelRegistryDep

logger = logging.getLogger(__name__)

models_router = APIRouter()


class ModelStatusResponse(BaseModel):
    model_id: str
    state: str
    error: str | None = None
    progress: dict | None = None


@models_router.get(
    "/models/{model_id}/status",
    response_model=ModelStatusResponse,
    summary="Get model load status",
    tags=["Models"],
)
async def get_model_status(
    model_id: str, model_registry: ModelRegistryDep
) -> ModelStatusResponse:
    state, error, progress = model_registry.get_status(model_id)
    logger.debug("Model status requested for %s: %s", model_id, state.name)
    return ModelStatusResponse(
        model_id=model_id,
        state=state.name.lower(),
        error=error,
        progress=progress,
    )
