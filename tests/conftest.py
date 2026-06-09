import asyncio
import logging
import os
import socket
import threading
from collections.abc import Generator
from pathlib import Path

import dotenv
import httpx
import pytest
import pytest_asyncio
import uvicorn
from models import MODELS, Model

from orchard.clients.client import Client
from orchard.engine.inference_engine import InferenceEngine
from orchard.server.app import create_app

logger = logging.getLogger(__name__)
dotenv.load_dotenv()

# These paths should be relative to the project root
PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / "logs_test"  # Use a separate log directory for tests

if LOG_DIR.exists():
    import shutil

    shutil.rmtree(LOG_DIR)

LOG_DIR.mkdir(parents=True, exist_ok=True)

SERVER_LOG_PATH = LOG_DIR / "python_server.test.log"
ENGINE_LOG_PATH = LOG_DIR / "engine.test.log"
CLIENT_LOG_PATH = LOG_DIR / "client.test.log"

ALL_MODELS = [m.checkpoint for m in MODELS]

SERVER_STARTUP_TIMEOUT_SECONDS = float(
    os.getenv("ORCHARD_TEST_SERVER_STARTUP_TIMEOUT_SECONDS", "120.0")
)

# Ensure we start with a clean slate in case a prior run crashed and left the engine up.
try:
    InferenceEngine.shutdown(timeout=30.0)
    logger.info("Pre-test engine cleanup complete.")
except RuntimeError as exc:
    raise RuntimeError(
        "Failed to stop existing engine before starting tests; manual cleanup required."
    ) from exc


@pytest.fixture(scope="session")
def engine() -> Generator[InferenceEngine, None, None]:
    """
    A session-scoped fixture that starts the PIE service using InferenceEngine,
    preloads models, and ensures clean shutdown.
    """
    logger.info("Setting up InferenceEngine for test session.")

    engine_instance: InferenceEngine | None = None
    try:
        engine_instance = InferenceEngine(
            client_log_file=CLIENT_LOG_PATH,
            engine_log_file=ENGINE_LOG_PATH,
            startup_timeout=120.0,
            load_models=ALL_MODELS,
        )
        logger.info("Local Engine is ready. Yielding engine instance.")
        yield engine_instance
    finally:
        if engine_instance is not None:
            engine_instance.close()
        logger.info("Test session finished. Local Engine shut down.")


@pytest.fixture(scope="session")
def client(engine: InferenceEngine) -> Generator[Client, None]:
    """
    Provides a synchronous Client instance connected to the shared engine.
    """
    client = engine.client()
    yield client
    client.close()


@pytest.fixture(params=MODELS, ids=lambda m: m.template_type)
def model(request: pytest.FixtureRequest) -> Model:
    return request.param


@pytest.fixture(params=MODELS, ids=lambda m: m.template_type)
def text_model_id(request: pytest.FixtureRequest) -> str:
    return request.param.checkpoint


@pytest.fixture(params=[m for m in MODELS if m.vision], ids=lambda m: m.template_type)
def vision_model_id(request: pytest.FixtureRequest) -> str:
    return request.param.checkpoint


@pytest.fixture
def moondream_model_id() -> str:
    return "moondream/moondream3-preview"


@pytest.fixture(params=MODELS, ids=lambda m: m.template_type)
def any_model_id(request: pytest.FixtureRequest) -> str:
    return request.param.checkpoint


@pytest_asyncio.fixture(scope="session")
async def live_server(engine: InferenceEngine):
    """
    Starts the FastAPI server against the shared test engine.
    """
    server_port = _free_local_port()
    fastapi_app = create_app(engine, close_engine_on_shutdown=False)
    config = uvicorn.Config(
        fastapi_app,
        host="127.0.0.1",
        port=server_port,
        log_level="warning",
        loop="asyncio",
        ws="none",
    )
    server = uvicorn.Server(config)
    server_url = f"http://localhost:{server_port}"
    server_errors: list[BaseException] = []

    def run_server() -> None:
        try:
            asyncio.run(server.serve())
        except BaseException as exc:
            server_errors.append(exc)

    server_thread = threading.Thread(
        target=run_server,
        name="orchard-test-fastapi",
        daemon=True,
    )
    server_thread.start()

    try:
        async with httpx.AsyncClient() as client:
            for _ in range(int(SERVER_STARTUP_TIMEOUT_SECONDS / 0.5)):
                if not server_thread.is_alive():
                    detail = f": {server_errors[0]!r}" if server_errors else ""
                    pytest.fail(f"FastAPI server exited prematurely{detail}.")
                try:
                    response = await client.get(f"{server_url}/health")
                except httpx.ConnectError:
                    await asyncio.sleep(0.5)
                    continue
                if response.status_code == 200:
                    logger.info("FastAPI live_server is up and healthy.")
                    break
                await asyncio.sleep(0.5)
            else:
                pytest.fail("Timeout waiting for FastAPI server to become connectable.")

        yield server_url

    finally:
        server.should_exit = True
        server_thread.join(timeout=10)
        if server_thread.is_alive():
            logger.warning("FastAPI server did not terminate cleanly; forcing exit.")
            server.force_exit = True
            server_thread.join(timeout=5)


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
