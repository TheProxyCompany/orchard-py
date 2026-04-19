from __future__ import annotations

import json
from pathlib import Path

from orchard.app.model_resolver import ModelResolver


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

    monkeypatch.setattr("orchard.app.model_resolver.snapshot_download", fake_snapshot_download)

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

    monkeypatch.setattr("orchard.app.model_resolver.snapshot_download", fake_snapshot_download)

    resolved = ModelResolver().resolve(repo_id)

    assert calls == [True, False]
    assert resolved.source == "hf_hub"
    assert resolved.model_path == downloaded.resolve()
