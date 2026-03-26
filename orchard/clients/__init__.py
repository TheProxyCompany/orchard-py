from orchard.app.ipc_dispatch import IPCState
from orchard.app.model_registry import ModelRegistry
from orchard.clients.client import Client
from orchard.clients.moondream import MoondreamClient


def get_client(
    model_id: str | None,
    ipc_state: IPCState,
    model_registry: ModelRegistry,
) -> Client:
    """Get a client for a given model ID."""
    if model_id == MoondreamClient.model_id:
        return MoondreamClient(ipc_state, model_registry)
    return Client(ipc_state, model_registry)


__all__ = ["Client", "MoondreamClient", "get_client"]
