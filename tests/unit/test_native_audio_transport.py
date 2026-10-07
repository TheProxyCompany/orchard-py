"""Transport lifecycle checks only; these are not media-model quality receipts."""

import asyncio
import sys
from pathlib import Path

import pytest

from orchard.clients.diarization import DiarizationSession
from orchard.clients.duplex import DuplexSession


@pytest.fixture
def audio_program(tmp_path: Path) -> Path:
    path = tmp_path / "native-audio-fixture"
    path.write_text(
        f"#!{sys.executable}\n"
        + r"""import json, sys
mode = sys.argv[sys.argv.index("--model") + 1]
def emit(**data):
    print(json.dumps(data), flush=True)
if mode == "malformed":
    print("[]", flush=True)
    sys.exit(0)
if mode == "startup_error":
    emit(type="error", message="Model assets unavailable")
    sys.exit(1)
emit(type="ready", frame_samples=1920, epoch=0)
for line in sys.stdin:
    command = json.loads(line)
    kind = command["type"]
    if kind == "close":
        break
    if kind == "interrupt":
        emit(type="audio", epoch=0, pcm=[0.1])
        emit(type="control_ack", id=command["id"], epoch=1)
        emit(type="audio", epoch=1, pcm=[0.2])
    elif kind == "reference":
        emit(type="request_error", id=command["id"], message="Speech reference belongs to an interrupted epoch")
    elif kind == "speak":
        for index in range(100):
            emit(type="audio", epoch=0, sequence=index, pcm=[0.1])
        emit(type="speech_done", epoch=0)
        emit(type="control_ack", id=command["id"], epoch=0)
    elif kind == "finish":
        emit(type="update", final=True, segments=[{"speaker_id":"session:speaker-1"}])
        emit(type="closed")
        break
    elif kind == "audio":
        emit(type="audio_ack", id=command["id"], epoch=0)
"""
    )
    path.chmod(0o700)
    return path


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode, exception, message",
    [
        ("malformed", TypeError, "JSON object"),
        ("startup_error", RuntimeError, "Model assets unavailable"),
    ],
)
async def test_startup_failure_closes_transport(
    audio_program, mode, exception, message
):
    with pytest.raises(exception, match=message):
        await DuplexSession.open(mode, binary=audio_program, ready_timeout=2)


@pytest.mark.asyncio
async def test_interrupt_discards_queued_old_audio_and_rejects_old_reference(
    audio_program,
):
    async with await DuplexSession.open("normal", binary=audio_program) as session:
        assert await session.interrupt() == 1
        stream = session.events()
        assert (await anext(stream))["type"] == "ready"
        assert (await anext(stream))["pcm"] == [0.2]
        with pytest.raises(RuntimeError, match="interrupted epoch"):
            await session.reference("stale fact", expected_epoch=0)


@pytest.mark.asyncio
async def test_slow_consumer_keeps_bounded_audio_and_completion(audio_program):
    async with await DuplexSession.open("normal", binary=audio_program) as session:
        await session.speak("bounded transport")
        assert len(session._events) <= 64
        assert any(event["type"] == "speech_done" for event in session._events)
        assert session._events[0]["type"] == "ready"
        with pytest.raises(ValueError, match="finite"):
            await session.push_audio(0, [float("nan")] * 1920)
    assert session._process.returncode == 0


@pytest.mark.asyncio
async def test_finish_drains_final_diarization_without_ack_race(audio_program):
    async with await DiarizationSession.open("normal", binary=audio_program) as session:
        await session.push_audio(0, [0.0] * 1920)
        await session.finish()

        async def collect():
            return [event async for event in session]

        events = await asyncio.wait_for(collect(), 2)
        assert any(event.get("final") and event["segments"] for event in events)
        assert events[-1]["type"] == "closed"
