import asyncio

import pytest

from orchard.app.ipc_dispatch import IPCState
from orchard.app.model_registry import (
    ModelEntry,
    ModelInfo,
    ModelLoadState,
    ModelRegistry,
)
from orchard.app.model_resolver import ResolvedModel
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


def test_handle_model_loaded_updates_minimum_memory_bytes():
    ctx = GlobalContext()
    ipc_state = IPCState(ctx)
    registry = ModelRegistry(ipc_state)
    canonical_id = "gemma/local"
    registry._alias_cache[canonical_id.lower()] = canonical_id
    registry._entries[canonical_id] = ModelEntry(
        state=ModelLoadState.ACTIVATING,
        info=ModelInfo(
            model_id=canonical_id,
            model_path="/tmp/gemma",
            formatter=object(),
        ),
    )

    registry.handle_model_loaded(
        {
            "event": "model_loaded",
            "model_id": canonical_id,
            "minimum_memory_bytes": 123456789,
        }
    )

    assert registry._entries[canonical_id].info is not None
    assert registry._entries[canonical_id].info.minimum_memory_bytes == 123456789


@pytest.mark.asyncio
async def test_schedule_model_uses_ready_alias_before_local_source_inspection(tmp_path):
    ctx = GlobalContext()
    ipc_state = IPCState(ctx)
    registry = ModelRegistry(ipc_state)
    model_path = tmp_path / "Model.GGUF"
    model_path.write_bytes(b"GGUF")
    canonical_id = "inspected-model"

    registry._alias_cache[str(model_path).lower()] = canonical_id
    registry._entries[canonical_id] = ModelEntry(
        state=ModelLoadState.READY,
        info=ModelInfo(
            model_id=canonical_id,
            model_path=str(model_path),
            formatter=object(),
        ),
    )

    state, resolved_id = await registry.schedule_model(str(model_path))

    assert state == ModelLoadState.READY
    assert resolved_id == canonical_id


@pytest.mark.asyncio
async def test_inspect_model_source_uses_path_cache(tmp_path):
    ctx = GlobalContext()
    ipc_state = IPCState(ctx)
    registry = ModelRegistry(ipc_state)
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"GGUF")
    inspected = ResolvedModel(
        canonical_id="inspected-model",
        model_path=model_path,
        source="local_source",
        formatter_config={"model_type": "llama"},
    )
    registry._local_source_inspection_cache[str(model_path.resolve())] = inspected

    resolved = await registry._inspect_model_source(
        str(model_path),
        ResolvedModel(
            canonical_id="model",
            model_path=model_path,
            source="local_source",
        ),
    )

    assert resolved is inspected


@pytest.mark.asyncio
async def test_get_info_uses_lowercase_alias_lookup(tmp_path):
    ctx = GlobalContext()
    ipc_state = IPCState(ctx)
    registry = ModelRegistry(ipc_state)
    requested_id = str(tmp_path / "Model.GGUF")
    canonical_id = "inspected-model"
    info = ModelInfo(
        model_id=canonical_id,
        model_path=requested_id,
        formatter=object(),
    )

    registry._alias_cache[requested_id.lower()] = canonical_id
    registry._entries[canonical_id] = ModelEntry(
        state=ModelLoadState.READY,
        info=info,
    )

    assert await registry.get_info(requested_id) is info
