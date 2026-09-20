"""Engine-free checks of the golden harness: sampling, drift reports, recording
and the one-model selection the recorder relies on."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from orchard.app.model_registry import ModelInfo
from orchard.clients.client import Client
from orchard.formatter.formatter import ChatFormatter
from tests.golden import golden_io
from tests.golden.cases.registry import GoldenCase, GreedyClient, run_cases
from tests.models import MATRIX, select_models

REPO = Path(__file__).resolve().parents[2]


# --- sampling -----------------------------------------------------------------


class _Registry:
    def __init__(self, info: ModelInfo) -> None:
        self._info = info

    async def get_info(self, model_id: str) -> ModelInfo:
        return self._info


def _client_for(tmp_path: Path, model_type: str) -> Client:
    """A real Client over a real Pantheon profile; rendering needs no engine."""
    model_path = tmp_path / model_type
    model_path.mkdir()
    (model_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    info = ModelInfo(
        model_id="m",
        model_path=str(model_path),
        formatter=ChatFormatter(str(model_path)),
    )
    return Client(None, _Registry(info))  # type: ignore[arg-type]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model_type", "recommended_temperature"),
    [("gpt_oss", 1.0), ("gemma4", 1.0), ("afmoe", 0.15)],
)
async def test_deterministic_alone_samples_but_the_golden_client_is_greedy(
    tmp_path: Path, model_type: str, recommended_temperature: float
) -> None:
    client = _client_for(tmp_path, model_type)

    plain = await client.arender_responses_prompt("m", input="hi", deterministic=True)
    assert plain["sampling_params"]["temperature"] == recommended_temperature
    assert plain["sampling_params"]["rng_seed"] == 11

    greedy = await GreedyClient(client).arender_responses_prompt(
        "m", input="hi", deterministic=True
    )
    assert greedy["sampling_params"]["temperature"] == 0.0
    assert greedy["sampling_params"]["deterministic"] is True
    assert greedy["sampling_params"]["rng_seed"] == 11
    assert greedy["rendered_prompt_text"] == plain["rendered_prompt_text"]


class _RecordingClient:
    images = object()

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def aresponses(self, model_id: str, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return "stream"


@pytest.mark.asyncio
async def test_the_runner_hands_every_case_a_greedy_client() -> None:
    inner = _RecordingClient()
    seen: list[Any] = []

    async def case(client: Any) -> None:
        seen.append(client)
        await client.aresponses("m", input="hi", stream=True, deterministic=True)
        await client.aresponses("m", input="hi", temperature=0.7)

    await GoldenCase("fake.case", case).run({"client": inner})

    assert isinstance(seen[0], GreedyClient)
    assert seen[0].images is inner.images  # modal calls pass through
    assert [(c["temperature"], c["deterministic"]) for c in inner.calls] == [
        (0.0, True),
        (0.0, True),
    ]
    assert inner.calls[0]["stream"] is True


# --- drift reports --------------------------------------------------------------


class _Token(BaseModel):
    type: str = "response.output_token"
    sequence_number: int
    token_id: int
    content: str


def _tokens(*pieces: str, ids: dict[str, int] | None = None) -> list[BaseModel]:
    return [
        _Token(sequence_number=n, token_id=(ids or {}).get(piece, n), content=piece)
        for n, piece in enumerate(pieces)
    ]


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setattr(golden_io, "DATA_DIR", tmp_path)
    monkeypatch.delenv("GOLDEN_RECORD", raising=False)
    monkeypatch.delenv("GOLDEN_ADD_VARIANT", raising=False)
    golden_io.discard_pending()
    yield tmp_path
    golden_io.discard_pending()


def _record(turns: dict[str, list[BaseModel]], scenario: str = "s") -> None:
    for turn, events in turns.items():
        golden_io.assert_or_record("arch", scenario, turn, events)
    golden_io.flush_pending()


def test_drift_report_shows_index_context_and_both_texts(data_dir: Path) -> None:
    _record({"turn1": _tokens("Thus", " output", ":", ' {"', "capital")})

    with pytest.raises(AssertionError) as raised:
        golden_io.assert_or_record(
            "arch", "s", "turn1", _tokens("Thus", " output", ":", " likely")
        )

    assert str(raised.value) == (
        "golden drift arch/s/turn1: event count: golden=5 live=4; "
        "first diff at index 3 after 'Thus output:':\n"
        "  golden: response.output_token ' {\"'\n"
        "  live:   response.output_token ' likely'"
    )


def test_drift_report_names_the_fields_when_the_text_is_the_same(
    data_dir: Path,
) -> None:
    _record({"turn1": _tokens("a", "b", ids={"b": 7})})

    with pytest.raises(AssertionError) as raised:
        golden_io.assert_or_record(
            "arch", "s", "turn1", _tokens("a", "b", ids={"b": 9})
        )

    assert "golden: response.output_token {'token_id': 7}" in str(raised.value)
    assert "live:   response.output_token {'token_id': 9}" in str(raised.value)


def test_a_drifted_turn_does_not_hide_the_next_one(data_dir: Path) -> None:
    _record(
        {"turn1": _tokens("a", "b"), "turn2": _tokens("c", "d"), "turn3": _tokens("e")}
    )
    reached: list[str] = []

    with pytest.raises(AssertionError) as raised, golden_io.collect_drift():
        golden_io.assert_or_record("arch", "s", "turn1", _tokens("a", "X"))
        reached.append("turn2")
        golden_io.assert_or_record("arch", "s", "turn2", _tokens("c", "Y"))
        reached.append("turn3")
        golden_io.assert_or_record("arch", "s", "turn3", _tokens("e"))

    assert reached == ["turn2", "turn3"]
    report = str(raised.value)
    assert report.count("golden drift arch/s/") == 2
    assert "arch/s/turn1" in report and "arch/s/turn2" in report
    assert "arch/s/turn3" not in report


def test_what_fails_a_case_is_unchanged(data_dir: Path) -> None:
    _record({"turn1": _tokens("a")})

    # Outside the runner a drift still raises at once.
    with pytest.raises(AssertionError, match="golden drift arch/s/turn1"):
        golden_io.assert_or_record("arch", "s", "turn1", _tokens("X"))

    # No drift: the case's own failure propagates untouched, and a clean case passes.
    with pytest.raises(KeyError), golden_io.collect_drift():
        golden_io.assert_or_record("arch", "s", "turn1", _tokens("a"))
        raise KeyError("semantic")
    with golden_io.collect_drift():
        golden_io.assert_or_record("arch", "s", "turn1", _tokens("a"))

    # Drift then a later failure: still one failure, reporting both.
    with (
        pytest.raises(AssertionError, match="golden drift") as raised,
        golden_io.collect_drift(),
    ):
        golden_io.assert_or_record("arch", "s", "turn1", _tokens("X"))
        raise KeyError("semantic")
    assert isinstance(raised.value.__cause__, KeyError)


@pytest.mark.asyncio
async def test_concurrent_cases_keep_their_own_drift(data_dir: Path) -> None:
    _record({"turn1": _tokens("a")}, scenario="drifts")
    _record({"turn1": _tokens("a")}, scenario="clean")

    async def drifts() -> None:
        golden_io.assert_or_record("arch", "drifts", "turn1", _tokens("X"))

    def clean() -> None:  # sync cases run in a thread
        golden_io.assert_or_record("arch", "clean", "turn1", _tokens("a"))

    def sync_drifts() -> None:
        golden_io.assert_or_record("arch", "drifts", "turn1", _tokens("Z"))

    failures = await run_cases(
        [
            GoldenCase("drifts", drifts),
            GoldenCase("clean", clean),
            GoldenCase("sync", sync_drifts),
        ],
        {},
    )

    assert [failure.case_id for failure in failures] == ["drifts", "sync"]
    assert "live:   response.output_token 'X'" in failures[0].detail
    assert "live:   response.output_token 'Z'" in failures[1].detail


# --- recording ------------------------------------------------------------------


def test_golden_record_replaces_the_file_only_when_asked(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _record({"turn1": _tokens("old"), "turn3": _tokens("stale")})
    path = data_dir / "arch" / "s.json"

    # Default: a recorded turn is asserted, a new turn is merged in.
    _record({"turn1": _tokens("old"), "turn2": _tokens("new")})
    assert sorted(json.loads(path.read_text())) == ["turn1", "turn2", "turn3"]

    monkeypatch.setenv("GOLDEN_RECORD", "1")
    golden_io.assert_or_record("arch", "s", "turn1", _tokens("fresh"))
    assert golden_io.pending_paths() == [path]
    assert json.loads(path.read_text())["turn1"][0]["content"] == "old"  # not yet

    golden_io.flush_pending()
    recorded = json.loads(path.read_text())
    assert list(recorded) == ["turn1"]  # turns the scenario no longer produces are gone
    assert recorded["turn1"][0]["content"] == "fresh"


# --- one-model selection ----------------------------------------------------------


def test_select_models() -> None:
    assert select_models(None) == MATRIX
    assert select_models(" ") == MATRIX
    assert [m.template_type for m in select_models("gpt_oss")] == ["gpt_oss"]
    # matrix order, whatever order the names come in
    assert [m.template_type for m in select_models("moondream3, gemma4")] == [
        "gemma4",
        "moondream3",
    ]
    with pytest.raises(ValueError, match="gpt_os"):
        select_models("gpt_os")


def _preloaded(models_env: str | None) -> subprocess.CompletedProcess[str]:
    """What tests/conftest.py would preload (its ALL_MODELS expression), in a
    fresh interpreter so the narrowing is read from the environment."""
    env = {k: v for k, v in os.environ.items() if k != "ORCHARD_TEST_MODELS"}
    if models_env is not None:
        env["ORCHARD_TEST_MODELS"] = models_env
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "import json; from tests.models import MODELS; "
            "print(json.dumps([m.checkpoint for m in MODELS]))",
        ],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_orchard_test_models_narrows_what_a_session_loads() -> None:
    assert json.loads(_preloaded("gpt_oss").stdout) == ["openai/gpt-oss-20b"]
    assert json.loads(_preloaded("gemma4,moondream3").stdout) == [
        "google/gemma-4-E2B-it",
        "moondream/moondream3-preview",
    ]
    assert json.loads(_preloaded(None).stdout) == [m.checkpoint for m in MATRIX]

    typo = _preloaded("gpt_os")
    assert typo.returncode != 0 and "unknown template type" in typo.stderr


@pytest.mark.parametrize(
    ("args", "env", "expected"),
    [
        ([], {}, "usage"),
        (["gpt_oss"], {"ORCHARD_CACHE_ROOT": "x"}, "PIE_LOCAL_BUILD"),
        (["gpt_oss"], {"PIE_LOCAL_BUILD": "x"}, "ORCHARD_CACHE_ROOT"),
    ],
)
def test_the_recorder_refuses_to_start_without_its_inputs(
    args: list[str], env: dict[str, str], expected: str
) -> None:
    base = {
        k: v
        for k, v in os.environ.items()
        if k not in {"PIE_LOCAL_BUILD", "ORCHARD_CACHE_ROOT"}
    }
    result = subprocess.run(
        ["bash", str(REPO / "scripts" / "record_golden.sh"), *args],
        env={**base, **env},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert expected in result.stderr
