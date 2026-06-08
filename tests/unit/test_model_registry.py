import asyncio
import json
import threading

import pytest

import orchard.app.model_registry as model_registry_module
from orchard.app.ipc_dispatch import IPCState
from orchard.app.model_registry import (
    MODEL_LOAD_CANCELLED,
    ModelEntry,
    ModelInfo,
    ModelLoadState,
    ModelRegistry,
)
from orchard.app.model_resolver import ResolvedModel
from orchard.engine.global_context import GlobalContext


class _FakeManagementSocket:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.sent: list[dict] = []

    async def asend(self, payload: bytes) -> None:
        self.sent.append(json.loads(payload.decode("utf-8")))

    async def arecv(self) -> bytes:
        return json.dumps(self.response).encode("utf-8")


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
async def test_schedule_model_builds_local_formatters_concurrently(
    monkeypatch, tmp_path
):
    ctx = GlobalContext()
    ipc_state = IPCState(ctx)
    registry = ModelRegistry(ipc_state)
    started: list[str] = []
    both_started = threading.Event()

    def fake_resolve(model_id: str) -> ResolvedModel:
        model_path = tmp_path / model_id
        model_path.mkdir()
        return ResolvedModel(
            canonical_id=model_id,
            model_path=model_path,
            source="hf_cache",
        )

    def fake_formatter(model_path: str) -> object:
        started.append(model_path)
        if len(started) == 2:
            both_started.set()
        assert both_started.wait(timeout=1.0)
        return object()

    monkeypatch.setattr(
        model_registry_module.ModelResolver,
        "resolve",
        lambda self, model_id: fake_resolve(model_id),
    )
    monkeypatch.setattr(model_registry_module, "ChatFormatter", fake_formatter)

    await asyncio.gather(
        registry.schedule_model("first"),
        registry.schedule_model("second"),
    )

    assert len(started) == 2


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


@pytest.mark.asyncio
async def test_cancel_activation_sends_cancel_model_load_and_fails_waiter():
    ctx = GlobalContext()
    ipc_state = IPCState(ctx)
    ipc_state.management_socket = _FakeManagementSocket({"status": "accepted"})
    registry = ModelRegistry(ipc_state)
    requested_id = "gemma4"
    canonical_id = "google/gemma-4-26B-A4B-it"
    loop = asyncio.get_running_loop()
    waiter = loop.create_future()

    registry._alias_cache[requested_id] = canonical_id
    registry._entries[canonical_id] = ModelEntry(
        state=ModelLoadState.ACTIVATING,
        info=ModelInfo(
            model_id=canonical_id,
            model_path="/tmp/gemma",
            formatter=object(),
        ),
        activation_future=waiter,
        activation_loop=loop,
    )

    response = await registry.cancel_activation(requested_id)
    await asyncio.sleep(0)

    assert response == {"status": "accepted"}
    assert ipc_state.management_socket.sent == [
        {
            "type": "cancel_model_load",
            "requested_id": requested_id,
            "canonical_id": canonical_id,
        }
    ]
    with pytest.raises(RuntimeError, match=MODEL_LOAD_CANCELLED):
        await waiter

    state, error, _ = registry.get_status(canonical_id)
    assert state == ModelLoadState.FAILED
    assert error == MODEL_LOAD_CANCELLED


@pytest.mark.asyncio
async def test_cancelled_activation_is_retryable(monkeypatch, tmp_path):
    ctx = GlobalContext()
    ipc_state = IPCState(ctx)
    registry = ModelRegistry(ipc_state)
    requested_id = "gemma4"
    canonical_id = "google/gemma-4-26B-A4B-it"
    model_path = tmp_path / "gemma"
    model_path.mkdir()

    registry._alias_cache[requested_id] = canonical_id
    registry._entries[canonical_id] = ModelEntry(
        state=ModelLoadState.FAILED,
        error=MODEL_LOAD_CANCELLED,
    )
    monkeypatch.setattr(
        registry._resolver,
        "resolve",
        lambda _: ResolvedModel(
            canonical_id=canonical_id,
            model_path=model_path,
            source="local",
        ),
    )
    monkeypatch.setattr(model_registry_module, "ChatFormatter", lambda _: object())

    state, resolved_id = await registry.schedule_model(requested_id)

    assert state == ModelLoadState.LOADING
    assert resolved_id == canonical_id
    assert registry._entries[canonical_id].error is None
