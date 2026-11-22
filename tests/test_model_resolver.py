import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from orchard.app.ipc_dispatch import IPCState
from orchard.app.model_registry import ModelRegistry
from orchard.app.model_resolver import ModelResolutionError, ModelResolver
from orchard.cli.preload_manifest import resolve_models
from orchard.engine.global_context import GlobalContext
from orchard.server.routes._common import resolve_model


def _write_hf_model_dir(model_dir: Path, *, model_type: str = "llama") -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": model_type, "hidden_size": 64}),
        encoding="utf-8",
    )
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    return model_dir


def test_resolves_relative_and_cwd_local_model_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    model_dir = _write_hf_model_dir(workspace / "models" / "tiny-llama")
    monkeypatch.chdir(workspace)

    resolver = ModelResolver()
    resolved = resolver.resolve("./models/tiny-llama")

    assert resolved.source == "local"
    assert resolved.model_path == model_dir.resolve()
    assert resolved.canonical_id == "tiny-llama"
    assert resolved.metadata["model_type"] == "llama"


def test_resolves_dot_when_cwd_is_model_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = _write_hf_model_dir(tmp_path / "direct-model")
    monkeypatch.chdir(model_dir)

    resolver = ModelResolver()
    resolved = resolver.resolve(".")

    assert resolved.source == "local"
    assert resolved.model_path == model_dir.resolve()
    assert resolved.canonical_id == "direct-model"


def test_resolves_diffusers_model_index_and_local_source_assets(
    tmp_path: Path,
) -> None:
    root = tmp_path / "allowed"
    diffusers_dir = root / "flux-model"
    diffusers_dir.mkdir(parents=True)
    (diffusers_dir / "model_index.json").write_text(
        json.dumps({"_class_name": "FluxPipeline"}),
        encoding="utf-8",
    )
    (diffusers_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    source_dir = root / "raw-source-dir"
    source_dir.mkdir(parents=True)
    source_file = root / "weights.gguf"
    source_file.write_bytes(b"gguf")

    resolver = ModelResolver(allowed_roots=[root])

    resolved_diffusers = resolver.resolve(str(diffusers_dir))
    assert resolved_diffusers.source == "local"
    assert resolved_diffusers.metadata["model_type"] == "flux"

    resolved_dir = resolver.resolve(str(source_dir))
    assert resolved_dir.source == "local_source"
    assert resolved_dir.canonical_id == "raw-source-dir"

    resolved_file = resolver.resolve(str(source_file))
    assert resolved_file.source == "local_source"
    assert resolved_file.canonical_id == "weights"


def test_resolves_models_under_orchard_model_dirs_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    root_a = tmp_path / "models_a"
    root_b = tmp_path / "models_b"
    model_b = _write_hf_model_dir(root_b / "qwen-local", model_type="qwen3")
    monkeypatch.setenv("ORCHARD_MODEL_DIRS", f"{root_a}{os.pathsep}{root_b}")

    resolver = ModelResolver()
    resolved = resolver.resolve(str(model_b))

    assert resolved.source == "local"
    assert resolved.model_path == model_b.resolve()


def test_preload_manifest_and_registry_allow_explicit_local_model_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    external_model = _write_hf_model_dir(tmp_path / "external" / "cli-model")
    entries = resolve_models([str(external_model)])

    assert len(entries) == 1
    assert entries[0].canonical_id == "cli-model"
    assert Path(entries[0].model_path) == external_model.resolve()

    registry = ModelRegistry(IPCState(GlobalContext()))
    with pytest.raises(ModelResolutionError, match="outside allowed model directories"):
        registry._resolver.resolve(str(external_model))

    registry.allow_local_path(str(external_model))
    resolved = registry._resolver.resolve(str(external_model))
    assert resolved.model_path == external_model.resolve()


def test_rejects_traversal_outside_allowed_roots_and_sibling_prefix_bypass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allowed = tmp_path / "models"
    allowed.mkdir()
    sibling_evil = _write_hf_model_dir(tmp_path / "models-evil" / "bad-model")
    outside_file = tmp_path / "outside-fixture.txt"
    outside_file.write_text("fixture", encoding="utf-8")
    monkeypatch.chdir(allowed)

    resolver = ModelResolver(allowed_roots=[allowed])

    with pytest.raises(ModelResolutionError, match="outside allowed model directories"):
        resolver.resolve("../outside-fixture.txt")

    with pytest.raises(ModelResolutionError, match="outside allowed model directories"):
        resolver.resolve(str(outside_file))

    with pytest.raises(ModelResolutionError, match="outside allowed model directories"):
        resolver.resolve(str(sibling_evil))

    with pytest.raises(ModelResolutionError, match="null bytes"):
        resolver.resolve("bad\x00model")


def test_rejects_symlink_escaping_allowed_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = _write_hf_model_dir(tmp_path / "outside-model")
    escape_link = allowed / "escaped-model"
    escape_link.symlink_to(outside)
    monkeypatch.chdir(allowed)

    resolver = ModelResolver(allowed_roots=[allowed])

    with pytest.raises(ModelResolutionError, match="outside allowed model directories"):
        resolver.resolve("./escaped-model")

    with pytest.raises(ModelResolutionError, match="outside allowed model directories"):
        resolver.resolve(str(escape_link))


def test_root_cwd_never_whitelists_entire_filesystem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside_model = _write_hf_model_dir(tmp_path / "untrusted-model")
    monkeypatch.setattr(os, "getcwd", lambda: os.sep)

    resolver = ModelResolver()
    with pytest.raises(ModelResolutionError, match="outside allowed model directories"):
        resolver.resolve(str(outside_model))


@pytest.mark.asyncio
async def test_route_resolve_model_returns_422_on_disallowed_local_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = _write_hf_model_dir(tmp_path / "outside-route-model")
    monkeypatch.chdir(allowed)

    registry = ModelRegistry(IPCState(GlobalContext()))
    with pytest.raises(HTTPException) as exc_info:
        await resolve_model(registry, str(outside))

    assert exc_info.value.status_code == 422
    detail = exc_info.value.detail
    assert isinstance(detail, dict)
    assert "outside allowed model directories" in detail["message"]


def test_hf_repo_id_falls_through_to_huggingface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cwd = tmp_path / "empty_cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    hf_snapshot = _write_hf_model_dir(tmp_path / "hf_snapshot", model_type="gemma4")
    (hf_snapshot / "model.safetensors").write_bytes(b"weights")

    resolver = ModelResolver()
    with patch(
        "orchard.app.model_resolver.snapshot_download",
        return_value=str(hf_snapshot),
    ) as mock_download:
        resolved = resolver.resolve("google/gemma-4-E2B-it")

    assert mock_download.called
    assert resolved.source == "hf_cache"
    assert resolved.canonical_id == "google/gemma-4-E2B-it"
    assert resolved.hf_repo == "google/gemma-4-E2B-it"
