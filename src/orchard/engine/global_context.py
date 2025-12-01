from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from proxy_inference_engine.app.model_registry import ModelRegistry

if TYPE_CHECKING:
    from proxy_inference_engine.app.ipc_dispatch import IPCState


@dataclass
class GlobalContext:
    """Global context for the inference engine."""

    ipc_state: IPCState | None = None
    model_registry: ModelRegistry | None = None
    dispatcher_thread: threading.Thread | None = None
    initialized: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)
    ref_count: int = 0
    last_telemetry: dict | None = None


# Holds per-process state for the inference engine.
global_context: GlobalContext = GlobalContext()
