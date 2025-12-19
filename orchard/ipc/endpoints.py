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
    else:
        # Default to the standard application cache directory.
        home = Path.home()
        mac_cache = home / "Library" / "Caches"
        base = (
            mac_cache if mac_cache.exists() and mac_cache.is_dir() else home / ".cache"
        )
        path = base / "com.theproxycompany" / "ipc"

    path.mkdir(parents=True, exist_ok=True)
    return path


def _as_ipc_url(path: Path) -> str:
    """Formats a filesystem path into an NNG ipc:// transport URL."""
    return f"ipc://{path.resolve()}"


# The root directory where all socket files will be created.
IPC_ROOT = _resolve_ipc_root()

# The endpoint for submitting inference and other requests to the engine.
# Pattern: PUSH/PULL (Many clients PUSH, one engine PULLs)
REQUEST_URL = _as_ipc_url(IPC_ROOT / "pie_requests.ipc")

# The endpoint for receiving responses and broadcast events from the engine.
# Pattern: PUB/SUB (One engine PUBlishes, many clients SUBscribe)
# Topics are used to route messages to the correct consumer.
RESPONSE_URL = _as_ipc_url(IPC_ROOT / "pie_responses.ipc")

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
]
