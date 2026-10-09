"""The exact NVIDIA Nemotron3 eight-speaker architecture through native PIE.

Speaker IDs are anonymous and session-local. Bind a Life Map entity only with
confirmed identity, preserving all overlapping tracks and model probabilities.
"""

from __future__ import annotations

from importlib.resources import files
from typing import Any

import yaml

from orchard.clients.duplex import _NativeAudioSession

DEFAULT_MODEL = "nvidia/Nemotron-3-Diarization"


def capabilities() -> dict[str, Any]:
    path = files("orchard").joinpath(
        "formatter/profiles/nemotron3_diarization/capabilities.yaml"
    )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class DiarizationSession(_NativeAudioSession):
    """Streaming probabilities and speaker segments from PIE-owned AOSC/FIFO state."""

    _program = "orchard-diarize"
    _environment = "ORCHARD_DIARIZATION_BINARY"
    _default_model = DEFAULT_MODEL

    async def finish(self) -> None:
        """Flush accepted audio; continue reading events through the final segments."""
        if self._done or self._closed:
            raise RuntimeError("Orchard diarization session is closed")
        assert self._process.stdin is not None
        async with self._write_lock:
            self._process.stdin.write(b'{"type":"finish"}\n')
            await self._process.stdin.drain()
