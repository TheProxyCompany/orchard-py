"""CLI helper for generating preload manifest files shared with the C++ engine."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from orchard.app.model_resolver import (
    ModelResolutionError,
    ModelResolver,
)


@dataclass(slots=True, frozen=True)
class PreloadEntry:
    requested_id: str
    canonical_id: str
    model_path: str


def resolve_models(
    models: Iterable[str],
    *,
    project_root: Path | None = None,
) -> list[PreloadEntry]:
    resolver = ModelResolver(project_root=project_root)
    entries: list[PreloadEntry] = []
    seen: set[tuple[str, str]] = set()

    for requested in models:
        try:
            resolved = resolver.resolve(requested)
        except ModelResolutionError as exc:  # pragma: no cover - defensive logging path
            raise SystemExit(
                f"Failed to resolve model '{requested}': {exc} (candidates={exc.candidates})"
            ) from exc
        entry_key = (resolved.canonical_id.lower(), str(resolved.model_path))
        if entry_key in seen:
            continue
        seen.add(entry_key)
        entries.append(
            PreloadEntry(
                requested_id=requested,
                canonical_id=resolved.canonical_id,
                model_path=str(resolved.model_path),
            )
        )
    return entries


def write_manifest(entries: list[PreloadEntry], output_path: Path) -> None:
    manifest = {"models": [asdict(entry) for entry in entries]}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a preload manifest consumed by orchard engine."
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination path for the manifest JSON file.",
    )
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        default=[],
        help="Model identifier to include. Repeat for multiple models.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Optional project root to seed ModelResolver lookup.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational stdout output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.models:
        raise SystemExit("At least one --model must be provided.")

    project_root = args.project_root
    if project_root is None:
        env_root = os.getenv("PIE_PROJECT_ROOT")
        if env_root:
            project_root = Path(env_root)

    entries = resolve_models(args.models, project_root=project_root)
    write_manifest(entries, args.output)

    if not args.quiet:
        print(f"Wrote manifest with {len(entries)} model(s) to {args.output}")
        for entry in entries:
            print(
                f"  {entry.requested_id} -> {entry.canonical_id} ({entry.model_path})"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
