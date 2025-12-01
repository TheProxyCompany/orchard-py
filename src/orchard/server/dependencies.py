from typing import Annotated

from fastapi import Depends, Request

from proxy_inference_engine.app.ipc_dispatch import IPCState
from proxy_inference_engine.app.model_registry import ModelRegistry


def get_ipc_state(request: Request) -> IPCState:
    """
    FastAPI dependency to get the single, process-wide IPCState instance.
    """
    return request.app.state.ipc_state


def get_model_registry(request: Request) -> ModelRegistry:
    """
    FastAPI dependency to get the single, process-wide ModelRegistry instance.
    """
    return request.app.state.model_registry


IPCStateDep = Annotated[IPCState, Depends(get_ipc_state)]
ModelRegistryDep = Annotated[ModelRegistry, Depends(get_model_registry)]
