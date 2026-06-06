from __future__ import annotations

import json
from pathlib import Path

from orchard.app.model_resolver import ModelResolver
from orchard.formatter.formatter import determine_template_type


def _write_common_model_files(model_dir: Path, *, repo_id: str) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "_name_or_path": repo_id,
                "model_type": "gemma4",
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")


def test_resolve_downloads_when_cached_snapshot_is_missing_a_shard(
    monkeypatch, tmp_path: Path
) -> None:
    repo_id = "google/gemma-4-31B-it"
    cached = tmp_path / "cached"
    downloaded = tmp_path / "downloaded"
    _write_common_model_files(cached, repo_id=repo_id)
    _write_common_model_files(downloaded, repo_id=repo_id)

    index_payload = {
        "weight_map": {
            "model.layers.0.weight": "model-00001-of-00002.safetensors",
            "model.layers.1.weight": "model-00002-of-00002.safetensors",
        }
    }
    for model_dir in (cached, downloaded):
        (model_dir / "model.safetensors.index.json").write_text(
            json.dumps(index_payload),
            encoding="utf-8",
        )

    (cached / "model-00002-of-00002.safetensors").write_bytes(b"cached-shard")
    (cached / "model-00001-of-00002.safetensors.incomplete").write_bytes(
        b"incomplete-shard"
    )
    (downloaded / "model-00001-of-00002.safetensors").write_bytes(b"shard-one")
    (downloaded / "model-00002-of-00002.safetensors").write_bytes(b"shard-two")

    calls: list[bool] = []

    def fake_snapshot_download(
        requested_repo_id: str, *, local_files_only: bool, allow_patterns: list[str]
    ) -> str:
        assert requested_repo_id == repo_id
        assert "model*.safetensors" in allow_patterns
        calls.append(local_files_only)
        return str(cached if local_files_only else downloaded)

    monkeypatch.setattr(
        "orchard.app.model_resolver.snapshot_download", fake_snapshot_download
    )

    resolved = ModelResolver().resolve(repo_id)

    assert calls == [True, False]
    assert resolved.source == "hf_hub"
    assert resolved.model_path == downloaded.resolve()


def test_resolve_downloads_when_cached_single_file_snapshot_is_broken(
    monkeypatch, tmp_path: Path
) -> None:
    repo_id = "google/gemma-4-E2B-it"
    cached = tmp_path / "cached"
    downloaded = tmp_path / "downloaded"
    _write_common_model_files(cached, repo_id=repo_id)
    _write_common_model_files(downloaded, repo_id=repo_id)

    (cached / "model.safetensors").symlink_to(cached / "missing-model.safetensors")
    (cached / "missing-model.safetensors.incomplete").write_bytes(b"incomplete")
    (downloaded / "model.safetensors").write_bytes(b"complete-model")

    calls: list[bool] = []

    def fake_snapshot_download(
        requested_repo_id: str, *, local_files_only: bool, allow_patterns: list[str]
    ) -> str:
        assert requested_repo_id == repo_id
        assert "model*.safetensors" in allow_patterns
        calls.append(local_files_only)
        return str(cached if local_files_only else downloaded)

    monkeypatch.setattr(
        "orchard.app.model_resolver.snapshot_download", fake_snapshot_download
    )

    resolved = ModelResolver().resolve(repo_id)

    assert calls == [True, False]
    assert resolved.source == "hf_hub"
    assert resolved.model_path == downloaded.resolve()


def test_resolve_accepts_nested_diffusers_safetensors_snapshot(
    monkeypatch, tmp_path: Path
) -> None:
    repo_id = "ideogram-ai/ideogram-4-fp8"
    cached = tmp_path / "cached"
    cached.mkdir(parents=True)
    (cached / "model_index.json").write_text(
        json.dumps({"_class_name": "Ideogram4Pipeline"}),
        encoding="utf-8",
    )
    for component, weight_name in (
        ("transformer", "diffusion_pytorch_model.safetensors"),
        ("unconditional_transformer", "diffusion_pytorch_model.safetensors"),
        ("text_encoder", "model.safetensors"),
        ("vae", "diffusion_pytorch_model.safetensors"),
    ):
        component_dir = cached / component
        component_dir.mkdir()
        (component_dir / "config.json").write_text("{}", encoding="utf-8")
        (component_dir / weight_name).write_bytes(b"weights")
        (component_dir / f"{weight_name}.index.json").write_text(
            json.dumps({"weight_map": {f"{component}.weight": weight_name}}),
            encoding="utf-8",
        )
    tokenizer_dir = cached / "tokenizer"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    calls: list[bool] = []

    def fake_snapshot_download(
        requested_repo_id: str, *, local_files_only: bool, allow_patterns: list[str]
    ) -> str:
        assert requested_repo_id == repo_id
        assert "model*.safetensors" in allow_patterns
        assert "*.safetensors" in allow_patterns
        calls.append(local_files_only)
        return str(cached)

    monkeypatch.setattr(
        "orchard.app.model_resolver.snapshot_download", fake_snapshot_download
    )

    resolved = ModelResolver().resolve(repo_id)

    assert calls == [True]
    assert resolved.source == "hf_cache"
    assert resolved.model_path == cached.resolve()
    assert resolved.formatter_config is not None
    assert resolved.formatter_config["model_type"] == "ideogram4"
    assert "template_type" not in resolved.formatter_config


def test_load_config_accepts_qwen_image_edit_model_index(tmp_path: Path) -> None:
    model_dir = tmp_path / "qwen-image-edit"
    model_dir.mkdir()
    (model_dir / "model_index.json").write_text(
        json.dumps({"_class_name": "QwenImageEditPipeline"}),
        encoding="utf-8",
    )

    config = ModelResolver._load_config(model_dir)  # noqa: SLF001

    assert config["model_type"] == "qwen_image_edit"
    assert config["source_format"] == "diffusers_directory"


def test_resolve_local_file_as_engine_inspected_source(tmp_path: Path) -> None:
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"GGUF")

    resolved = ModelResolver().resolve(str(model_file))

    assert resolved.source == "local_source"
    assert resolved.canonical_id == "model"
    assert resolved.model_path == model_file.resolve()
    assert resolved.formatter_config is None


def test_hf_resolved_model_passes_repo_hint_to_formatter_config(tmp_path: Path) -> None:
    repo_id = "meta-llama/Llama-3.1-8B-Instruct"
    model_dir = tmp_path / "downloaded"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "llama"}),
        encoding="utf-8",
    )
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    resolved = ModelResolver()._build_resolved_model(  # noqa: SLF001
        model_dir,
        source="hf_cache",
        canonical_id=repo_id,
        hf_repo=repo_id,
    )

    assert resolved.formatter_config is not None
    assert resolved.formatter_config["model_type"] == "llama"
    assert resolved.formatter_config["_name_or_path"] == repo_id
    assert determine_template_type(resolved.formatter_config) == "default"


def test_resolve_parakeet_tdt_config_uses_audio_profile(tmp_path: Path) -> None:
    repo_id = "mlx-community/parakeet-tdt-0.6b-v3"
    model_dir = tmp_path / "downloaded"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "_name_or_path": repo_id,
                "target": "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel",
                "model_defaults": {"tdt_durations": [0, 1, 2, 3, 4]},
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "model.safetensors").write_bytes(b"weights")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    resolved = ModelResolver()._build_resolved_model(  # noqa: SLF001
        model_dir,
        source="hf_cache",
        canonical_id=repo_id,
        hf_repo=repo_id,
    )

    assert resolved.formatter_config is not None
    assert resolved.metadata["model_type"] == "parakeet_tdt"
    assert resolved.formatter_config["model_type"] == "parakeet_tdt"
    assert "template_type" not in resolved.formatter_config
    assert resolved.formatter_config["_name_or_path"] == repo_id
