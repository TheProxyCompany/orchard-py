import io
import re
import struct
import wave
from collections.abc import Sequence

import pytest

from orchard.clients.client import Client

pytestmark = pytest.mark.asyncio

QWEN3_TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
PARAKEET_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
QWEN3_ASR_MODEL = "Qwen/Qwen3-ASR-0.6B"
PHRASE = "hello this is a test"
TARGET_SAMPLE_RATE = 16_000


def _normalize_transcript(text: str) -> str:
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _resample_linear(samples: list[float], source_rate: int, target_rate: int) -> list[float]:
    if source_rate == target_rate or not samples:
        return samples
    target_count = max(1, round(len(samples) * target_rate / source_rate))
    if target_count == 1:
        return [samples[0]]
    output: list[float] = []
    ratio = source_rate / target_rate
    last = len(samples) - 1
    for index in range(target_count):
        position = index * ratio
        left = min(int(position), last)
        right = min(left + 1, last)
        mix = position - left
        output.append(samples[left] * (1.0 - mix) + samples[right] * mix)
    return output


def _wav_to_float32_pcm(wav_bytes: bytes, target_rate: int) -> list[float]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        source_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())

    assert channels >= 1, "TTS WAV must have at least one channel"
    assert sample_width == 2, f"expected 16-bit PCM WAV, got sample_width={sample_width}"
    values = struct.unpack(f"<{len(frames) // sample_width}h", frames)
    if channels == 1:
        mono = [sample / 32768.0 for sample in values]
    else:
        mono = [
            sum(values[index + channel] for channel in range(channels))
            / (channels * 32768.0)
            for index in range(0, len(values), channels)
        ]
    return _resample_linear(mono, source_rate, target_rate)


async def _synthesize_phrase(client: Client) -> list[float]:
    artifacts = await client.audio.agenerate(
        QWEN3_TTS_MODEL,
        PHRASE,
        language="English",
        speaker="Aiden",
        sample_rate=24_000,
        max_output_tokens=128,
        temperature=0.9,
        top_k=50,
        seed=1337,
        deterministic=True,
    )
    assert isinstance(artifacts, Sequence)
    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.type == "audio"
    assert artifact.mime_type == "audio/wav"
    assert artifact.data.startswith(b"RIFF")
    assert len(artifact.data) > 44
    return _wav_to_float32_pcm(artifact.data, TARGET_SAMPLE_RATE)


@pytest.mark.parametrize(
    ("stt_label", "stt_model"),
    [
        ("parakeet", PARAKEET_MODEL),
        ("qwen3_asr", QWEN3_ASR_MODEL),
    ],
)
async def test_qwen3_tts_to_speech_to_text_transcription(
    client: Client, stt_label: str, stt_model: str
):
    print(f"\n\033[1;33m━━━ qwen3_tts · telephone → {stt_label} ━━━\033[0m", flush=True)
    pcm = await _synthesize_phrase(client)
    assert pcm, "TTS produced no audio samples"

    transcript = await client.audio.atranscribe(stt_model, pcm)
    normalized = _normalize_transcript(transcript)
    print(f"\ntranscript={transcript!r} normalized={normalized!r}", flush=True)
    assert normalized == PHRASE
