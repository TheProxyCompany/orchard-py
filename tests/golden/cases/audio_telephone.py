import io
import re
import struct
import wave
from collections.abc import Sequence

import pytest

from orchard.clients.client import Client

pytestmark = pytest.mark.asyncio

TTS_MODELS = [
    (
        "qwen3_tts_0_6b",
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        {"language": "English", "speaker": "Aiden", "temperature": 0.9, "top_k": 50},
    ),
    (
        "qwen3_tts_1_7b",
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        {"language": "English", "speaker": "Aiden", "temperature": 0.9, "top_k": 50},
    ),
]
PARAKEET_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
STT_MODELS = [
    ("parakeet", PARAKEET_MODEL),
    ("qwen3_asr_0_6b", "Qwen/Qwen3-ASR-0.6B"),
    ("qwen3_asr_1_7b", "Qwen/Qwen3-ASR-1.7B"),
]
PHRASES = [
    "hello this is a test",
    "the quick brown fox jumps over the lazy dog",
    "today we test local speech in a quiet room",
    "set the kitchen timer for tomorrow morning after breakfast",
    "proxy orchard handles audio images and text together",
    "blue square red circle green triangle",
]
TARGET_SAMPLE_RATE = 16_000


def _normalize_transcript(text: str) -> str:
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _resample_linear(
    samples: list[float], source_rate: int, target_rate: int
) -> list[float]:
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
    assert sample_width == 2, (
        f"expected 16-bit PCM WAV, got sample_width={sample_width}"
    )
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


async def _synthesize_phrase(
    client: Client, model_id: str, phrase: str, options: dict[str, object]
) -> list[float]:
    artifacts = await client.audio.agenerate(
        model_id,
        phrase,
        sample_rate=24_000,
        max_output_tokens=128,
        seed=1337,
        deterministic=True,
        **options,
    )
    assert isinstance(artifacts, Sequence)
    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.type == "audio"
    assert artifact.mime_type == "audio/wav"
    assert artifact.data.startswith(b"RIFF")
    assert len(artifact.data) > 44
    return _wav_to_float32_pcm(artifact.data, TARGET_SAMPLE_RATE)


async def test_tts_to_speech_to_text_transcription(client: Client):
    # Every (tts_model, phrase) leg is independent, and within a leg the STT
    # models are independent given the synthesized pcm. Run legs concurrently
    # instead of 12 serial synth calls + 36 serial transcriptions. Cap 2:
    # this suite shares the engine with a chat suite and the image chain —
    # wider audio fan-out contributed to a GPU-watchdog engine death
    # (2026-07-08 storm, exit 134).
    import asyncio

    gate = asyncio.Semaphore(2)

    async def leg(tts_label, tts_model, tts_options, phrase):
        async with gate:
            print(
                f"\n\033[1;33m━━━ {tts_label} · telephone · {phrase!r} ━━━\033[0m",
                flush=True,
            )
            pcm = await _synthesize_phrase(client, tts_model, phrase, tts_options)
            assert pcm, "TTS produced no audio samples"

            async def check(stt_label, stt_model):
                transcript = await client.audio.atranscribe(stt_model, pcm)
                normalized = _normalize_transcript(transcript)
                print(
                    f"\n{tts_label} → {stt_label}: "
                    f"transcript={transcript!r} normalized={normalized!r}",
                    flush=True,
                )
                assert normalized == phrase

            await asyncio.gather(
                *(check(stt_label, stt_model) for stt_label, stt_model in STT_MODELS)
            )

    await asyncio.gather(
        *(
            leg(tts_label, tts_model, tts_options, phrase)
            for tts_label, tts_model, tts_options in TTS_MODELS
            for phrase in PHRASES
        )
    )
