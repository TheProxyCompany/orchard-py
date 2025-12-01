import asyncio
import atexit
import logging
import subprocess
import sys
from collections.abc import Generator
from pathlib import Path

import httpx
import pytest
import pytest_asyncio

from proxy_inference_engine.clients.client import Client
from proxy_inference_engine.engine.inference_engine import InferenceEngine

logger = logging.getLogger(__name__)

# These paths should be relative to the project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
RELEASE_DIR = PROJECT_ROOT / "release"
ENGINE_EXE = RELEASE_DIR / "bin/pie_engine"
SERVER_EXE = RELEASE_DIR / "bin/pie_server"
LOG_DIR = PROJECT_ROOT / "logs_test"  # Use a separate log directory for tests

if LOG_DIR.exists():
    import shutil

    shutil.rmtree(LOG_DIR)

LOG_DIR.mkdir(parents=True, exist_ok=True)

SERVER_LOG_PATH = LOG_DIR / "python_server.test.log"
ENGINE_LOG_PATH = LOG_DIR / "engine.test.log"
CLIENT_LOG_PATH = LOG_DIR / "client.test.log"

MODEL_IDS = [
    "llama-3.1-8b-instruct",
    "moondream3",
    # "gemma-3-27b",
]

SERVER_PORT = 8001

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
            load_models=MODEL_IDS,
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


@pytest_asyncio.fixture(scope="session")
async def live_server():
    """
    Starts the FastAPI server in a separate process.
    """
    if not SERVER_EXE.exists():
        pytest.fail(f"Server executable not found at {SERVER_EXE}.")

    def server_cleanup():
        logger.info("Server cleanup started.")
        if server_proc and server_proc.poll() is None:
            server_proc.terminate()
            server_proc.wait(timeout=10)
        logger.info("Server cleanup complete.")

    atexit.register(server_cleanup)

    server_cmd = [
        sys.executable,
        "-m",
        "proxy_inference_engine.cli.main",
        "serve",
        f"--port={SERVER_PORT}",
        "--engine-log-file",
        str(ENGINE_LOG_PATH),
        "--models",
        *MODEL_IDS,
    ]

    server_proc = None

    try:
        with open(SERVER_LOG_PATH, "wb") as log_file:
            server_proc = subprocess.Popen(
                server_cmd,
                stdout=log_file,
                stderr=log_file,
            )

        # Wait for the server to be connectable
        server_url = f"http://localhost:{SERVER_PORT}"
        async with httpx.AsyncClient() as client:
            for _ in range(30):
                try:
                    await client.get(f"{server_url}/health")
                    logger.info("FastAPI live_server is up and healthy.")
                    break
                except httpx.ConnectError:
                    if server_proc.poll() is not None:
                        pytest.fail(
                            f"FastAPI server exited prematurely. Check logs at {SERVER_LOG_PATH}"
                        )
                    await asyncio.sleep(0.5)
            else:
                pytest.fail("Timeout waiting for FastAPI server to become connectable.")

        yield server_url

    finally:
        atexit.unregister(server_cleanup)
        server_cleanup()
