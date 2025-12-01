import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from proxy_inference_engine.engine.inference_engine import InferenceEngine
from proxy_inference_engine.server.exceptions import InferenceError
from proxy_inference_engine.server.routes.chat import chat_router
from proxy_inference_engine.server.routes.completions import completions_router
from proxy_inference_engine.server.routes.embeddings import embeddings_router
from proxy_inference_engine.server.routes.models import models_router
from proxy_inference_engine.server.routes.responses import responses_router

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "DEBUG"),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def create_app(inference_engine: InferenceEngine) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        app.state.inference_engine = inference_engine
        app.state.client = inference_engine.client()
        app.state.ipc_state = inference_engine.ipc_state()
        app.state.model_registry = inference_engine.model_registry()
        logger.info("FastAPI server lifespan started.")
        yield
        logger.info("FastAPI server lifespan shutting down.")
        app.state.client.close()
        app.state.inference_engine.close()
        logger.info("Application shutdown complete.")

    app = FastAPI(
        title="Proxy Inference Engine Server",
        description="API server interfacing with the Proxy Inference Engine C++ backend.",
        version="0.0.1",
        lifespan=lifespan,
    )

    @app.exception_handler(InferenceError)
    async def inference_exception_handler(request: Request, exc: InferenceError):
        logger.error(f"Caught InferenceError: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": f"Inference operation failed: {exc}"},
        )

    @app.get("/health", tags=["Health"], summary="Service health check")
    async def health_check() -> dict[str, str]:
        return {"status": "ok"}

    logger.info("Including routers...")
    app.include_router(completions_router, prefix="/v1", tags=["Completions"])
    app.include_router(chat_router, prefix="/v1", tags=["Chat"])
    app.include_router(embeddings_router, prefix="/v1", tags=["Embeddings"])
    app.include_router(responses_router, prefix="/v1", tags=["Responses"])
    app.include_router(models_router, prefix="/v1")
    logger.info("Routers included.")

    return app
