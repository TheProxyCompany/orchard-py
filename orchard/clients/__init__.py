from orchard.app.ipc_dispatch import IPCState
from orchard.app.model_registry import ModelRegistry
from orchard.clients.client import Client
from orchard.clients.moondream import MoondreamClient
from orchard.clients.privacy_filter import OpenAIPrivacyFilterClient


def get_client(
    model_id: str | None,
    ipc_state: IPCState,
    model_registry: ModelRegistry,
) -> Client:
    """Get a client for a given model ID."""
    if model_id == MoondreamClient.model_id:
        return MoondreamClient(ipc_state, model_registry)
    if model_id == OpenAIPrivacyFilterClient.model_id:
        return OpenAIPrivacyFilterClient(ipc_state, model_registry)
    return Client(ipc_state, model_registry)


__all__ = ["Client", "MoondreamClient", "OpenAIPrivacyFilterClient", "get_client"]
