from __future__ import annotations

import os
from pathlib import Path


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
    path.mkdir(parents=True, exist_ok=True)
    return path


# Unix socket paths are limited to 103 bytes on macOS. The engine
# (pie::utils::bounded_ipc_root, platform_utils.cpp) maps any IPC root whose
# longest socket path would not fit onto /tmp/orchard-ipc-<uid>-<fnv1a64 of the
# canonical root>. This mirrors that rule exactly so both sides meet on the
# same sockets: a private cache root under $TMPDIR (what the test session and
# containerized runs use) is already too long for the direct path.
_IPC_SOCKET_PATH_MAX_BYTES = 103
_LONGEST_SOCKET_NAME = "pie_response_ffffffffffffffff.ipc"


def _fnv1a64(value: str) -> int:
    digest = 0xCBF29CE484222325
    for byte in value.encode("utf-8"):
        digest ^= byte
        digest = (digest * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return digest


def _bounded_ipc_root(candidate: Path) -> Path:
    canonical = candidate.resolve()
    if (
        len(str(canonical / _LONGEST_SOCKET_NAME).encode("utf-8"))
        <= _IPC_SOCKET_PATH_MAX_BYTES
    ):
        return canonical
    compact = (
        Path("/tmp").resolve()
        / f"orchard-ipc-{os.getuid()}-{_fnv1a64(str(canonical)):016x}"
    )
    if (
        len(str(compact / _LONGEST_SOCKET_NAME).encode("utf-8"))
        > _IPC_SOCKET_PATH_MAX_BYTES
    ):
        raise RuntimeError(f"Could not construct a bounded IPC root for {canonical}")
    return compact


def _as_ipc_url(path: Path) -> str:
    """Formats a filesystem path into an NNG ipc:// transport URL."""
    return f"ipc://{path.resolve()}"


# The root directory where all socket files will be created.
IPC_ROOT = _resolve_ipc_root()

# The endpoint for submitting inference and other requests to the engine.
# Pattern: PUSH/PULL (Many clients PUSH, one engine PULLs)
REQUEST_URL = _as_ipc_url(IPC_ROOT / "pie_requests.ipc")

# The endpoint for broadcast events from the engine. An engine that predates
# response_pull_path also answers requests here.
# Pattern: PUB/SUB (One engine PUBlishes, many clients SUBscribe)
# Topics are used to route messages to the correct consumer. A subscriber that
# falls about a thousand messages behind loses the oldest ones, with no error
# on either side.
RESPONSE_URL = _as_ipc_url(IPC_ROOT / "pie_responses.ipc")


def response_pull_path(response_channel_id: int) -> Path:
    """The socket file this client listens on for its own response deltas.
    Pattern: PUSH/PULL (the engine PUSHes, this client PULLs). Requests ask for
    this route (serialization.RESPONSE_TRANSPORT); the engine dials the file on
    the first delta it has for the channel, and a client that falls behind
    makes the engine wait instead of losing messages."""
    return IPC_ROOT / f"pie_response_{response_channel_id:x}.ipc"


# The endpoint for synchronous management commands (e.g., load_model).
# Pattern: REQ/REP (One client sends a REQ, one engine sends a REP)
MANAGEMENT_URL = _as_ipc_url(IPC_ROOT / "pie_management.ipc")

# --- Topic Prefixes for the PUB/SUB Channel ---

# Topic prefix for response deltas targeted at a specific client.
# A client subscribes to b_RESPONSE_TOPIC_PREFIX + its_channel_id_hex.
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
    "response_pull_path",
]
