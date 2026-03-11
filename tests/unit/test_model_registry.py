import asyncio

import pytest

from orchard.app.ipc_dispatch import IPCState
from orchard.app.model_registry import (
    ModelEntry,
    ModelInfo,
    ModelLoadState,
    ModelRegistry,
)
from orchard.engine.global_context import GlobalContext


@pytest.mark.asyncio
async def test_ensure_loaded_raises_on_model_load_failed_event(monkeypatch):
    ctx = GlobalContext()
    ipc_state = IPCState(ctx)
    registry = ModelRegistry(ipc_state)
    canonical_id = "broken/model"
    info = ModelInfo(
        model_id=canonical_id,
        model_path="/tmp/broken-model",
        formatter=object(),
    )

    async def fake_schedule_model(
        requested_model_id: str, *, force_reload: bool = False
    ) -> tuple[ModelLoadState, str]:
        del requested_model_id, force_reload
        registry._alias_cache[canonical_id.lower()] = canonical_id
        if canonical_id not in registry._entries:
            entry = ModelEntry(state=ModelLoadState.LOADING)
            entry.info = info
            entry.event.set()
            registry._entries[canonical_id] = entry
        return ModelLoadState.LOADING, canonical_id

    async def fake_await_model(
        model_id: str, timeout: float | None = None
    ) -> tuple[ModelLoadState, ModelInfo | None, str | None]:
        del model_id, timeout
        return ModelLoadState.LOADING, info, None

    async def fake_send_load_model_command(
        *, requested_id: str, canonical_id: str, info: ModelInfo
    ) -> None:
        del requested_id, info
        asyncio.get_running_loop().call_soon(
            registry.handle_model_load_failed,
            {
                "event": "model_load_failed",
                "model_id": canonical_id,
                "error": "missing shard",
            },
        )

    monkeypatch.setattr(registry, "schedule_model", fake_schedule_model)
    monkeypatch.setattr(registry, "await_model", fake_await_model)
    monkeypatch.setattr(
        registry, "_send_load_model_command", fake_send_load_model_command
    )

    with pytest.raises(RuntimeError, match="missing shard"):
        await registry.ensure_loaded(canonical_id)

    state, error, _ = registry.get_status(canonical_id)
    assert state == ModelLoadState.FAILED
    assert error == "missing shard"
