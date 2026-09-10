from __future__ import annotations

import os
import sys
from pathlib import Path

_MAX_SOCKET_PATH_BYTES = 103
_LONGEST_SOCKET_NAME = "pie_response_ffffffffffffffff.ipc"


def _canonical_path(path: Path) -> Path:
    # Match weakly_canonical: resolve the existing prefix, then normalize the suffix.
    try:
        suffix = []
        prefix = path
        while not prefix.exists() and prefix.parent != prefix:
            suffix.append(prefix.name)
            prefix = prefix.parent
        return Path(
            os.path.normpath(prefix.resolve(strict=True).joinpath(*reversed(suffix)))
        )
    except (OSError, RuntimeError):
        try:
            return Path(os.path.abspath(path))
        except OSError:
            return Path(os.path.normpath(path))


def _bounded_ipc_root(candidate: Path) -> Path:
    # Keep the path projection identical to PIE's platform_utils.cpp.
    canonical = _canonical_path(candidate)
    if len(os.fsencode(canonical / _LONGEST_SOCKET_NAME)) <= _MAX_SOCKET_PATH_BYTES:
        return canonical

    digest = 0xCBF29CE484222325
    for byte in os.fsencode(canonical):
        digest = ((digest ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    compact = _canonical_path(Path("/tmp")) / f"orchard-ipc-{os.getuid()}-{digest:016x}"
    if len(os.fsencode(compact / _LONGEST_SOCKET_NAME)) > _MAX_SOCKET_PATH_BYTES:
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
        path = Path(ipc_root_env)
    elif cache_root_env := os.getenv("ORCHARD_CACHE_ROOT"):
        path = Path(cache_root_env) / "ipc"
    else:
        # Default to the standard application cache directory.
        home = Path.home()
        if sys.platform == "darwin":
            base = home / "Library" / "Caches"
        else:
            base = Path(os.getenv("XDG_CACHE_HOME") or home / ".cache")
        path = base / "com.theproxycompany" / "ipc"

    if not ipc_root_env:
        # PIE creates its cache directory before weakly canonicalizing the IPC child.
        path.parent.mkdir(parents=True, exist_ok=True)
    path = _bounded_ipc_root(path)
    try:
        path.mkdir(mode=0o700, parents=True)
    except FileExistsError:
        if not path.is_dir():
            raise
    else:
        path.chmod(0o700)
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
