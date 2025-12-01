import logging
import os
import time
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from pynng.nng import pynng

logger = logging.getLogger(__name__)

from proxy_inference_engine.app.ipc_dispatch import IPCState
from proxy_inference_engine.ipc import endpoints as ipc_endpoints


@dataclass(frozen=True)
class EnginePaths:
    cache_dir: Path
    ready_file: Path
    pid_file: Path
    refs_file: Path
    lock_file: Path
    client_log_file: Path
    engine_log_file: Path

def get_engine_file_paths(client_log_file: Path | None, engine_log_file: Path | None) -> EnginePaths:
    cache_dir = cache_root()
    if client_log_file and not client_log_file.exists():
        client_log_file.parent.mkdir(parents=True, exist_ok=True)
    if engine_log_file and not engine_log_file.exists():
        engine_log_file.parent.mkdir(parents=True, exist_ok=True)
    return EnginePaths(
        cache_dir=cache_dir,
        ready_file=cache_dir / "engine.ready",
        pid_file=cache_dir / "engine.pid",
        refs_file=cache_dir / "engine.refs",
        lock_file=cache_dir / "engine.lock",
        client_log_file=client_log_file if client_log_file else cache_dir / "client.log",
        engine_log_file=engine_log_file if engine_log_file else cache_dir / "engine.log",
    )


def cache_root() -> Path:
    home = Path.home()
    mac_cache = home / "Library" / "Caches"
    base = mac_cache if mac_cache.exists() else home / ".cache"
    target = base / "com.proxycompany.pie"
    target.mkdir(parents=True, exist_ok=True)
    return target


def locate_engine_binary() -> Path:
    try:
        package_root = Path(str(resources.files("proxy_inference_engine")))
    except ModuleNotFoundError as exc:  # pragma: no cover - import-time failure
        raise FileNotFoundError("proxy_inference_engine package is unavailable.") from exc

    candidate = (package_root.parent / "bin" / "pie_engine").resolve()
    if candidate.exists() and os.access(candidate, os.X_OK):
        return candidate

    # Dev fallback
    dev_candidate = Path(__file__).resolve().parents[3] / "release" / "bin" / "pie_engine"
    if dev_candidate.exists() and os.access(dev_candidate, os.X_OK):
        return dev_candidate

    raise FileNotFoundError(
        f"pie_engine binary not found at expected production path {candidate} or dev path {dev_candidate}."
        " Build artifacts before using InferenceEngine."
    )


def dial_with_retry(socket: pynng.Socket, url: str, attempts: int = 5, delay: float = 0.2) -> None:
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            socket.dial(url, block=True)
            return
        except Exception as exc:  # pragma: no cover - best-effort retry
            logger.debug(
                "Failed to connect to IPC endpoint %s: %s (attempt %d)",
                url,
                exc,
                attempt,
            )
            last_exc = exc
            time.sleep(delay)
    error_msg = f"Failed to connect to IPC endpoint {url} after {attempts} attempts."
    raise RuntimeError(error_msg) from last_exc


def close_sockets(ipc_state: IPCState | None) -> None:
    if not ipc_state:
        return

    if ipc_state.request_socket:
        ipc_state.request_socket.close()
        ipc_state.request_socket = None
    if ipc_state.response_socket:
        ipc_state.response_socket.close()
        ipc_state.response_socket = None
    if ipc_state.management_socket:
        ipc_state.management_socket.close()
        ipc_state.management_socket = None


def initialize_sockets(
    ipc_state: IPCState,
    response_channel_id: int,
) -> None:
    ipc_state.request_socket = pynng.Push0()
    ipc_state.request_socket.recv_max_size = 0
    dial_with_retry(ipc_state.request_socket, ipc_endpoints.REQUEST_URL)
    ipc_state.response_socket = pynng.Sub0()
    ipc_state.response_socket.recv_max_size = 0
    ipc_state.response_socket.subscribe(f"resp:{response_channel_id:x}".encode("ascii"))
    ipc_state.response_socket.subscribe(ipc_endpoints.EVENT_TOPIC_PREFIX)
    dial_with_retry(ipc_state.response_socket, ipc_endpoints.RESPONSE_URL)
    ipc_state.management_socket = pynng.Req0()
    ipc_state.management_socket.recv_max_size = 0
    dial_with_retry(ipc_state.management_socket, ipc_endpoints.MANAGEMENT_URL)
