"""Golden-stream snapshots for the golden test tier.

The first clean run of a (template_type, scenario, turn) records the model's
exact normalized stream as ground truth; every later run asserts against it.
Drift in event order, deltas, field_path, counts, or usage fails the run against
the model's own recorded truth.

Snapshots are sharded one file per (template_type, scenario), model-first, at
``data/<template_type>/<scenario>.json`` (each file holds ``{turn: events}``).
Model-first so a regression's blast radius is one folder, recording a new model
never touches another's file, and parallel runs don't clobber a shared file.

The human is in the loop at the git diff: after a recording run, review the new
file and either bless it (commit) or, if it captured a bug, delete it and
re-record once the engine is fixed.

Identifiers (``resp_…``, ``msg_…``, ``call_…``) are random per run, so they are
canonicalized to stable first-seen tokens — this pins event-to-event references
(a delta's ``item_id`` still has to match its function call's ``id``) without
pinning the random value. Timestamps are dropped. Everything behavioral
(``type``, ``sequence_number``, ``output_index``, ``delta``, ``field_path``,
``arguments``, ``text``, ``status``, usage counts) is pinned exactly.
"""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

DATA_DIR = Path(__file__).parent / "data"

ID_KEYS = frozenset(
    {"id", "item_id", "call_id", "response_id"}
)  # random per run -> canonicalized
TIMESTAMP_KEYS = frozenset({"created_at", "completed_at"})  # wall clock -> dropped

# Baselines staged this test, written to disk only if the whole test passes
# (flush_pending / discard_pending, driven by the golden conftest). A failing
# test never persists a buggy golden.
_pending: dict[Path, dict[str, list[dict]]] = {}


def normalize(events: list[BaseModel]) -> list[dict]:
    """Normalize one turn's raw event stream into its run-invariant snapshot.

    Maps each random id to a stable first-seen token (``msg_0``, ``call_0``) so
    cross-references stay pinned without pinning the value, and nulls timestamps.
    """
    ids: dict[str, str] = {}
    counts: dict[str, int] = {}

    def token(value: str) -> str:
        if value not in ids:
            prefix = value.split("_", 1)[0] if "_" in value else "id"
            ids[value] = f"{prefix}_{counts.get(prefix, 0)}"
            counts[prefix] = counts.get(prefix, 0) + 1
        return ids[value]

    def canon(node: Any) -> Any:
        if isinstance(node, dict):
            return {
                key: None
                if key in TIMESTAMP_KEYS
                else token(value)
                if key in ID_KEYS and isinstance(value, str)
                else canon(value)
                for key, value in node.items()
            }
        if isinstance(node, list):
            return [canon(item) for item in node]
        return node

    return [canon(event.model_dump(mode="json")) for event in events]


def assert_or_record(
    template_type: str,
    scenario: str,
    turn: str,
    events: list[BaseModel],
) -> None:
    """Assert this turn's stream against its golden, or stage it if absent.

    Missing ``data/<template_type>/<scenario>.json`` turn → STAGE the current
    normalized stream; it is written to disk only if the whole test passes (so a
    failing test never records a buggy baseline). Present → assert exact
    equality; any drift raises with the first differing event.
    """
    live = normalize(events)
    path = DATA_DIR / template_type / f"{scenario}.json"
    text = path.read_text().strip() if path.exists() else ""
    data = json.loads(text) if text else {}

    recorded = data.get(turn)
    if recorded is None:
        _pending.setdefault(path, {})[turn] = live
        print(
            f"\n\033[1;36m[golden] staged baseline "
            f"{template_type}/{scenario}/{turn} ({len(live)} events) "
            f"— written only if the test passes\033[0m",
            flush=True,
        )
        return

    if recorded == live:
        return

    if len(recorded) != len(live):
        detail = f"event count: golden={len(recorded)} live={len(live)}"
    else:
        detail = "event count matches but contents differ"
    for i, (exp, act) in enumerate(zip(recorded, live, strict=False)):
        if exp != act:
            detail += f"; first diff at index {i}:\n  golden: {exp}\n  live:   {act}"
            break
    else:
        n = min(len(recorded), len(live))
        if len(live) > n:
            detail += f"; live has an extra event at index {n}:\n  live: {live[n]}"
        elif len(recorded) > n:
            detail += (
                f"; live is missing the event at index {n}:\n  golden: {recorded[n]}"
            )

    raise AssertionError(f"golden drift {template_type}/{scenario}/{turn}: {detail}")


def flush_pending() -> None:
    """Write the baselines staged this test to disk. Call only after it passed."""
    for path, turns in _pending.items():
        text = path.read_text().strip() if path.exists() else ""
        data = json.loads(text) if text else {}
        data.update(turns)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    _pending.clear()


def pending_paths() -> list[Path]:
    """Golden files staged for recording but not yet flushed."""
    return sorted(_pending)


def discard_pending() -> None:
    """Drop the baselines staged this test. Call when it failed (or to reset)."""
    _pending.clear()
