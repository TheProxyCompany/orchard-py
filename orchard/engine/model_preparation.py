"""Load descriptors through their existing adapters, without opening sessions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orchard.clients.diarization import DiarizationSession
from orchard.clients.diarization import capabilities as speaker_capabilities
from orchard.clients.duplex import DuplexSession
from orchard.clients.duplex import capabilities as voice_capabilities


@dataclass(frozen=True)
class ModelLoadRequest:
    """The same options must be supplied when opening the eventual session."""

    model_id: str
    options: dict[str, Any] | None = None

    def key(self) -> tuple[str, str]:
        return self.model_id, json.dumps(self.options, sort_keys=True, allow_nan=False)


def native_adapter(model_id: str) -> type[DuplexSession | DiarizationSession] | None:
    """Native source knowledge stays beside the source adapters, not the host."""
    descriptor = Path(model_id).expanduser() / "config.json"
    if descriptor.is_file():
        value = json.loads(descriptor.read_text(encoding="utf-8"))
        if value.get("model_type") == "moshi":
            return DuplexSession
        if value.get("model_type") == "nemotron3_diarization":
            return DiarizationSession
    voice = voice_capabilities()["duplex"]
    if model_id in voice["checkpoints"]:
        return DuplexSession
    # Local native asset directories are accepted by the same Rust adapter.
    if Path(model_id).is_dir() and all(
        (Path(model_id) / voice[key]).is_file()
        for key in ("model_file", "codec_file", "tokenizer_file")
    ):
        return DuplexSession
    speakers = speaker_capabilities()["diarization"]
    if model_id == speakers["model_id"] or (
        Path(model_id).is_file() and Path(model_id).name == speakers["model_file"]
    ):
        return DiarizationSession
    return None
