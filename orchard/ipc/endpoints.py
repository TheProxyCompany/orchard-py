from __future__ import annotations

import os
from pathlib import Path

_IPC_SOCKET_PATH_MAX_BYTES = 103
_LONGEST_SOCKET_NAME = "pie_response_ffffffffffffffff.ipc"
_FNV1A64_OFFSET_BASIS = 0xCBF29CE484222325
_FNV1A64_PRIME = 0x100000001B3


def _fnv1a64(value: bytes) -> int:
    digest = _FNV1A64_OFFSET_BASIS
    for byte in value:
        digest ^= byte
        digest = (digest * _FNV1A64_PRIME) & 0xFFFFFFFFFFFFFFFF
    return digest


def _socket_path_fits(path: Path) -> bool:
    return len(os.fsencode(path)) <= _IPC_SOCKET_PATH_MAX_BYTES


def _bounded_ipc_root(candidate: Path) -> Path:
    canonical = candidate.expanduser().resolve()
    if _socket_path_fits(canonical / _LONGEST_SOCKET_NAME):
        return canonical

    digest = _fnv1a64(os.fsencode(canonical))
    compact = Path("/tmp").resolve() / f"orchard-ipc-{os.getuid()}-{digest:016x}"
    if not _socket_path_fits(compact / _LONGEST_SOCKET_NAME):
        raise RuntimeError(f"Could not construct a bounded IPC root for {canonical}")
    return compact


def _resolve_ipc_root() -> Path:
    """
    Determines the stable, user-specific root directory for IPC socket files.
    This ensures that all Orchard processes communicate through a predictable,
    private location, avoiding pollution of system-wide directories like /tmp.
    """
    # ORCHARD_IPC_ROOT is an escape hatch for development or containerized environments.
    if ipc_root_env := os.getenv("ORCHARD_IPC_ROOT"):
        path = Path(ipc_root_env).expanduser().resolve()
    elif cache_root_env := os.getenv("ORCHARD_CACHE_ROOT"):
        path = Path(cache_root_env).expanduser().resolve() / "ipc"
    else:
        # Default to the standard application cache directory.
        home = Path.home()
        mac_cache = home / "Library" / "Caches"
        base = (
            mac_cache if mac_cache.exists() and mac_cache.is_dir() else home / ".cache"
        )
        path = base / "com.theproxycompany" / "ipc"

    path = _bounded_ipc_root(path)
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.environ["ORCHARD_IPC_ROOT"] = os.fspath(path)
    return path


def _as_ipc_url(path: Path) -> str:
    """Formats a filesystem path into an NNG ipc:// transport URL."""
    return f"ipc://{path.resolve()}"


# The root directory where all socket files will be created.
IPC_ROOT = _resolve_ipc_root()

# The endpoint for submitting inference and other requests to the engine.
# Pattern: PUSH/PULL (Many clients PUSH, one engine PULLs)
REQUEST_URL = _as_ipc_url(IPC_ROOT / "pie_requests.ipc")

# The endpoint for broadcast events and legacy response deltas from the engine.
# Pattern: PUB/SUB (One engine PUBlishes, many clients SUBscribe)
RESPONSE_URL = _as_ipc_url(IPC_ROOT / "pie_responses.ipc")


def response_route_path(response_channel_id: int) -> Path:
    """Return the dedicated flow-controlled response endpoint for one client."""
    return IPC_ROOT / f"pie_response_{response_channel_id:x}.ipc"


def response_route_url(response_channel_id: int) -> str:
    return _as_ipc_url(response_route_path(response_channel_id))


# The endpoint for synchronous management commands (e.g., load_model).
# Pattern: REQ/REP (One client sends a REQ, one engine sends a REP)
MANAGEMENT_URL = _as_ipc_url(IPC_ROOT / "pie_management.ipc")

# --- Topic Prefixes for the PUB/SUB Channel ---

# Topic prefix for legacy response deltas targeted at a specific client.
# New clients keep this subscription for compatibility with pre-pull_v1 engines.
RESPONSE_TOPIC_PREFIX = b"resp:"

# Topic prefix for global, broadcast events (e.g., engine_ready).
# Clients subscribe to this prefix to receive all system-wide notifications.
EVENT_TOPIC_PREFIX = b"__PIE_EVENT__:"

__all__ = [
    "EVENT_TOPIC_PREFIX",
    "IPC_ROOT",
    "MANAGEMENT_URL",
    "REQUEST_URL",
    "RESPONSE_TOPIC_PREFIX",
    "RESPONSE_URL",
    "response_route_path",
    "response_route_url",
]
