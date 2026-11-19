import json
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from orchard.app.model_registry import ModelLoadState, ModelRegistry
from orchard.app.model_resolver import ModelResolutionError, ModelResolver
from orchard.server.app import create_app
from orchard.server.routes._common import resolve_model


def test_app_canonicalizes_configured_model_directories(tmp_path):
    directory = tmp_path / "models"
    directory.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(directory, target_is_directory=True)
    assert create_app(Mock(), model_directories=directory).state.model_directories == (
        directory,
    )
    assert create_app(
        Mock(), model_directories=[alias, directory]
    ).state.model_directories == (directory,)


@pytest.mark.asyncio
async def test_http_uses_operator_registered_local_model(tmp_path, monkeypatch):
    checkpoint = tmp_path / "local-checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"model_type":"llama"}')
    (checkpoint / "tokenizer.json").write_text("{}")
    monkeypatch.setattr("orchard.app.model_registry.ChatFormatter", Mock())
    registry = ModelRegistry(Mock())
    state, name = await registry.schedule_model(str(checkpoint))
    assert state == ModelLoadState.LOADING
    assert name == checkpoint.name
    registry._entries[name].state = ModelLoadState.READY
    for identifier in [str(checkpoint), name]:
        canonical, info = await resolve_model(registry, identifier)
        assert canonical == name
        assert info.model_path == str(checkpoint)


@pytest.mark.asyncio
async def test_http_rejects_unregistered_absolute_path(tmp_path, monkeypatch):
    checkpoint = tmp_path / "unregistered"
    checkpoint.mkdir()
    download = Mock()
    monkeypatch.setattr("orchard.app.model_resolver.snapshot_download", download)
    registry = ModelRegistry(Mock())
    with pytest.raises(HTTPException) as error:
        await resolve_model(registry, str(checkpoint))
    assert error.value.status_code == 422
    download.assert_not_called()


def test_http_hub_name_uses_snapshot_even_when_local_names_match(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    shadow = tmp_path / "org" / "checkpoint"
    shadow.mkdir(parents=True)
    (shadow / "config.json").write_text('{"model_type":"wrong"}')
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    config = {"model_type": "moondream3", "auto_map": {"AutoConfig": "config.Config"}}
    (snapshot / "config.json").write_text(json.dumps(config))
    (snapshot / "tokenizer.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"fixture")
    download = Mock(return_value=str(snapshot))
    monkeypatch.setattr("orchard.app.model_resolver.snapshot_download", download)
    source_config = Mock()
    monkeypatch.setattr(ModelResolver, "_load_python_config", source_config)
    resolved = ModelResolver().resolve_hub_model("org/checkpoint")
    assert resolved.model_path == snapshot
    assert resolved.canonical_id == "org/checkpoint"
    download.assert_called_once()
    source_config.assert_not_called()


@pytest.mark.asyncio
async def test_served_directories_are_resolved_per_app_and_preserve_registered_aliases(
    tmp_path, monkeypatch
):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    checkpoint = second / "local-model"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text('{"model_type":"llama"}')
    (checkpoint / "tokenizer.json").write_text("{}")
    monkeypatch.setattr("orchard.app.model_registry.ChatFormatter", Mock())
    registry = ModelRegistry(Mock())
    state, name = await registry.schedule_http_model("local-model", (first, second))
    assert state == ModelLoadState.LOADING and name == "local-model"
    registry._entries[name].state = ModelLoadState.READY
    assert (await resolve_model(registry, str(checkpoint), (second,)))[
        1
    ].model_path == str(checkpoint)
    with pytest.raises(HTTPException) as error:
        await resolve_model(registry, name, (first,))
    assert error.value.status_code == 422
    await registry.schedule_model(str(checkpoint))
    assert (await resolve_model(registry, name))[0] == name


def test_served_model_references_stay_in_configured_directories(tmp_path, monkeypatch):
    models = tmp_path / "models"
    blobs = tmp_path / "blobs"
    checkpoint = models / "sample"
    checkpoint.mkdir(parents=True)
    blobs.mkdir()
    (checkpoint / "config.json").write_text('{"model_type":"llama"}')
    (checkpoint / "tokenizer.json").write_text("{}")
    shard = blobs / "weights"
    shard.write_bytes(b"fixture")
    link = checkpoint / "model.safetensors"
    link.symlink_to(shard)
    alias = models / "external"
    alias.symlink_to(blobs, target_is_directory=True)
    download = Mock()
    monkeypatch.setattr("orchard.app.model_resolver.snapshot_download", download)
    resolver = ModelResolver()
    for identifier in ("sample", "external", "../blobs", str(blobs)):
        with pytest.raises(ModelResolutionError):
            resolver.resolve_http_model(identifier, (models,))
    assert (
        resolver.resolve_http_model("sample", (models, blobs)).model_path == checkpoint
    )
    link.unlink()
    index = checkpoint / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"layer": str(shard)}}))
    with pytest.raises(ModelResolutionError):
        resolver.resolve_http_model("sample", (models,))
    assert resolver.resolve_http_model("sample", (models, blobs)).source == "local"
    download.assert_not_called()
