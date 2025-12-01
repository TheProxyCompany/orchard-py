from proxy_inference_engine.app.ipc_dispatch import IPCState
from proxy_inference_engine.app.model_registry import ModelRegistry
from proxy_inference_engine.clients.client import Client
from proxy_inference_engine.clients.moondream import MoondreamClient


def get_client(
    model_id: str | None,
    ipc_state: IPCState,
    model_registry: ModelRegistry,
) -> Client:
    """Get a client for a given model ID."""
    if model_id == MoondreamClient.model_id:
        return MoondreamClient(ipc_state, model_registry)
    else:
        return Client(ipc_state, model_registry)


__all__ = ["Client", "MoondreamClient", "get_client"]
