"""CPU-only facade tests; no actual engine, model, microphone, or GPU is used."""

import asyncio
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from orchard.clients.diarization import DiarizationSession
from orchard.clients.duplex import DuplexSession
from orchard.engine.global_context import global_context
from orchard.engine.inference_engine import InferenceEngine
from orchard.engine.model_preparation import ModelLoadRequest, native_adapter


@pytest.mark.asyncio
async def test_all_five_start_concurrently_and_native_assets_never_go_to_chat_resolver(
    monkeypatch,
):
    model_ids = [
        "google/gemma-4-12B-it",
        "kyutai/moshika-rag-candle-bf16",
        "moondream/moondream3-preview",
        "nvidia/Nemotron-3-Diarization",
        "mlx-community/parakeet-tdt-0.6b-v3",
    ]
    entered = []
    release = asyncio.Event()

    async def regular(model):
        assert model not in (model_ids[1], model_ids[3]), (
            "Native assets require their source adapter"
        )
        entered.append(model)
        await release.wait()

    async def native(model, *, options=None):
        entered.append(model)
        await release.wait()

    monkeypatch.setattr(
        global_context, "model_registry", SimpleNamespace(ensure_loaded=regular)
    )
    monkeypatch.setattr(DuplexSession, "prepare", native)
    monkeypatch.setattr(DiarizationSession, "prepare", native)
    engine = InferenceEngine.__new__(InferenceEngine)
    task = asyncio.create_task(engine.load_models([*model_ids, model_ids[1]]))
    try:
        async with asyncio.timeout(2):
            while len(entered) < 5:
                await asyncio.sleep(0)
        assert sorted(entered) == sorted(model_ids)
        assert not task.done(), "Scheduling is not a readiness receipt"
    finally:
        release.set()
        await task


@pytest.mark.asyncio
async def test_option_specific_preloads_and_sessions_share_the_same_native_options(
    tmp_path, monkeypatch
):
    program = tmp_path / "native-prepare"
    record = tmp_path / "commands.jsonl"
    program.write_text(
        f"#!{sys.executable}\n"
        + r"""
import json, os, sys
model = sys.argv[sys.argv.index('--model') + 1]
options = json.loads(sys.argv[sys.argv.index('--options-json') + 1])
with open(os.environ['PREPARATION_RECORD'], 'a') as output:
    output.write(json.dumps({'model': model, 'options': options, 'prepare': '--prepare-only' in sys.argv}) + '\n')
if '--prepare-only' in sys.argv:
    print(json.dumps({'type': 'prepared', 'model_id': model, 'runtime_model_id': model}), flush=True)
else:
    print(json.dumps({'type': 'ready', 'frame_samples': 1920, 'epoch': 0}), flush=True)
    for line in sys.stdin:
        if json.loads(line)['type'] == 'close': break
"""
    )
    program.chmod(0o700)
    monkeypatch.setenv("PREPARATION_RECORD", str(record))
    monkeypatch.setenv("ORCHARD_DIARIZATION_BINARY", str(program))
    monkeypatch.setattr(
        global_context, "model_registry", SimpleNamespace(ensure_loaded=AsyncMock())
    )
    engine = InferenceEngine.__new__(InferenceEngine)
    model = "nvidia/Nemotron-3-Diarization"
    options = {"device": "metal", "chunk_frames": 8}
    await engine.load_models(
        [ModelLoadRequest(model, options), ModelLoadRequest(model, {"device": "cpu"})]
    )
    async with await DiarizationSession.open(model, options=options):
        pass
    commands = [json.loads(line) for line in record.read_text().splitlines()]
    assert len(commands) == 3, (
        "Distinct residency options cannot be deduplicated by model name"
    )
    prepared = next(
        command
        for command in commands
        if command["prepare"] and command["options"]["device"] == "metal"
    )
    opened = next(command for command in commands if not command["prepare"])
    assert prepared["options"] == opened["options"] == options
    assert DiarizationSession.effective_options(None)["device"] == (
        "metal" if sys.platform == "darwin" else "cpu"
    )


def test_local_alternate_descriptors_select_the_native_adapter_without_a_repo_allowlist(
    tmp_path,
):
    (tmp_path / "config.json").write_text('{"model_type":"moshi","rag":true}')
    assert native_adapter(str(tmp_path)) is DuplexSession
    (tmp_path / "config.json").write_text('{"model_type":"nemotron3_diarization"}')
    assert native_adapter(str(tmp_path)) is DiarizationSession
    assert native_adapter("unknown/ordinary-model") is None


@pytest.mark.asyncio
async def test_native_prepare_requires_confirmed_matching_receipt(tmp_path):
    program = tmp_path / "bad-preparation"
    program.write_text(
        f'#!{sys.executable}\nprint(\'{{"type":"ready","model_id":"model"}}\')\n'
    )
    program.chmod(0o700)
    with pytest.raises(RuntimeError, match="matching residency receipt"):
        await DuplexSession.prepare("model", binary=program)
