"""Model resolution utilities for mapping user-friendly model identifiers to local or cached model assets."""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

__all__ = [
    "ModelCatalogEntry",
    "ModelResolutionError",
    "ModelResolver",
    "ResolvedModel",
]


@dataclass(slots=True)
class ModelCatalogEntry:
    canonical_id: str
    model_path: Path
    source: str  # local | hf_cache | hf_hub
    hf_repo: str | None
    tokens: set[str]
    aliases: set[str]
    metadata: dict[str, str] = field(default_factory=dict)

    def to_resolved(self) -> ResolvedModel:
        return ResolvedModel(
            canonical_id=self.canonical_id,
            model_path=self.model_path,
            source=self.source,
            metadata=self.metadata,
            hf_repo=self.hf_repo,
        )


@dataclass(slots=True)
class ResolvedModel:
    canonical_id: str
    model_path: Path
    source: str
    metadata: dict[str, str]
    hf_repo: str | None


class ModelResolutionError(RuntimeError):
    """Raised when a model identifier cannot be resolved or is ambiguous."""

    def __init__(self, message: str, candidates: Sequence[str] | None = None):
        super().__init__(message)
        self.candidates = list(candidates or [])


class ModelResolver:
    """Resolves model identifiers to local filesystem paths with fuzzy matching support."""

    def __init__(self, project_root: Path | None = None):
        base_candidates: list[Path] = []
        if project_root is not None:
            base_candidates.append(Path(project_root))

        env_project = os.getenv("PIE_PROJECT_ROOT")
        if env_project:
            base_candidates.append(Path(env_project).expanduser())

        meipass_dir = getattr(sys, "_MEIPASS", None)
        if meipass_dir:
            base_candidates.append(Path(meipass_dir))

        base_candidates.append(Path(__file__).resolve().parents[3])
        base_candidates.append(Path.cwd())

        resolved_project = None
        for candidate in base_candidates:
            try:
                root = candidate.resolve()
            except OSError:
                continue
            if (root / ".models").exists():
                resolved_project = root
                break

        if resolved_project is None:
            resolved_project = base_candidates[0].resolve()

        self._project_root = resolved_project

        models_env = os.getenv("PIE_MODELS_PATH")
        if models_env:
            self._local_models_dir = Path(models_env).expanduser().resolve()
        else:
            self._local_models_dir = self._project_root / ".models"
            if not self._local_models_dir.exists():
                alt_candidates = [Path.cwd() / ".models"]
                for alt in alt_candidates:
                    if alt.exists():
                        self._local_models_dir = alt.resolve()
                        break

        self._entries_by_canonical: dict[str, ModelCatalogEntry] = {}
        self._entries_by_path: dict[Path, ModelCatalogEntry] = {}
        self._alias_map: dict[str, ModelCatalogEntry] = {}

        self._load_local_models()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def resolve(self, requested_id: str) -> ResolvedModel:
        identifier = requested_id.strip()
        if not identifier:
            raise ModelResolutionError("Model identifier cannot be empty.")

        # 1. Direct filesystem path (relative or absolute)
        direct_entry = self._try_direct_path(identifier)
        if direct_entry is not None:
            return direct_entry.to_resolved()

        # 2. Exact alias/canonical match
        normalized = identifier.lower()
        if normalized in self._alias_map:
            return self._alias_map[normalized].to_resolved()
        if identifier in self._entries_by_canonical:
            return self._entries_by_canonical[identifier].to_resolved()

        # 3. Fuzzy local matching
        matches = self._score_local_entries(identifier)
        if len(matches) == 1:
            return matches[0].to_resolved()
        if len(matches) > 1:
            raise ModelResolutionError(
                f"Model identifier '{requested_id}' is ambiguous.",
                candidates=[entry.canonical_id for entry in matches],
            )

        # 4. Try Hugging Face cache / hub for explicit repo IDs
        if "/" in identifier or identifier.count("--") >= 1:
            entry = self._resolve_huggingface_repo(identifier)
            return entry.to_resolved()

        raise ModelResolutionError(
            f"Unable to resolve model identifier '{requested_id}'.",
        )

    def list_entries(self) -> list[ModelCatalogEntry]:
        return list(self._entries_by_canonical.values())

    def lookup_alias(self, alias: str) -> ModelCatalogEntry | None:
        return self._alias_map.get(alias.lower())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _try_direct_path(self, identifier: str) -> ModelCatalogEntry | None:
        candidates: list[Path] = []
        raw_path = Path(identifier)
        if raw_path.is_absolute() and raw_path.exists():
            candidates.append(raw_path)
        else:
            rel_to_project = (self._project_root / identifier).resolve()
            if rel_to_project.exists():
                candidates.append(rel_to_project)
            elif raw_path.exists():  # relative to CWD
                candidates.append(raw_path.resolve())

        for path in candidates:
            if path in self._entries_by_path:
                return self._entries_by_path[path]
            if path.is_dir():
                entry = self._register_directory(path, source="local")
                return entry
        return None

    def _score_local_entries(self, identifier: str) -> list[ModelCatalogEntry]:
        if not self._entries_by_canonical:
            return []
        query_tokens = self._tokenize(identifier)
        if not query_tokens:
            return []

        scored: list[tuple[int, ModelCatalogEntry]] = []
        for entry in self._entries_by_canonical.values():
            score = len(query_tokens & entry.tokens)
            if score > 0:
                scored.append((score, entry))

        if not scored:
            return []

        scored.sort(key=lambda item: item[0], reverse=True)
        top_score = scored[0][0]
        top_entries = [entry for score, entry in scored if score == top_score]
        return top_entries

    def _resolve_huggingface_repo(self, repo_id: str) -> ModelCatalogEntry:
        allow_patterns = [
            "*.json",
            "model*.safetensors",
            "*.py",
            "tokenizer.model",
            "*.tiktoken",
            "tiktoken.model",
            "*.txt",
            "*.jsonl",
            "*.jinja",
        ]
        try:
            path = Path(
                snapshot_download(
                    repo_id, local_files_only=True, allow_patterns=allow_patterns
                )
            )
            source = "hf_cache"
        except LocalEntryNotFoundError:
            path = Path(
                snapshot_download(
                    repo_id, local_files_only=False, allow_patterns=allow_patterns
                )
            )
            source = "hf_hub"
        entry = self._register_directory(
            path, canonical_id=repo_id, source=source, hf_repo=repo_id
        )
        return entry

    def _register_directory(
        self,
        directory: Path,
        *,
        canonical_id: str | None = None,
        source: str,
        hf_repo: str | None = None,
    ) -> ModelCatalogEntry:
        directory = directory.resolve()
        if directory in self._entries_by_path:
            entry = self._entries_by_path[directory]
            # Update source/hf_repo if we learned more
            if hf_repo and not entry.hf_repo:
                entry.hf_repo = hf_repo
            return entry

        config = self._load_config(directory)
        canonical = canonical_id or self._determine_canonical_id(config, directory)
        inferred_repo = hf_repo or self._infer_hf_repo(config)
        metadata = self._collect_metadata(config)
        aliases = self._build_aliases(canonical, directory.name, inferred_repo)
        tokens = self._build_tokens(canonical, directory.name, inferred_repo, metadata)

        entry = ModelCatalogEntry(
            canonical_id=canonical,
            model_path=directory,
            source=source,
            hf_repo=inferred_repo,
            tokens=tokens,
            aliases=aliases,
            metadata=metadata,
        )

        self._entries_by_canonical[canonical] = entry
        self._entries_by_path[directory] = entry
        for alias in aliases:
            normalized = alias.lower()
            # Preserve first registration for deterministic behaviour
            self._alias_map.setdefault(normalized, entry)
        return entry

    def _load_local_models(self) -> None:
        if not self._local_models_dir.exists():
            return
        for child in sorted(self._local_models_dir.iterdir()):
            if child.is_dir():
                try:
                    self._register_directory(child, source="local")
                except Exception:
                    # Skip directories that do not contain valid model artifacts
                    continue

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_config(model_dir: Path) -> dict:
        config_file = model_dir / "config.json"
        if not config_file.exists():
            raise ModelResolutionError(
                f"Model directory '{model_dir}' is missing config.json."
            )
        with config_file.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _determine_canonical_id(config: dict, model_dir: Path) -> str:
        name_or_path = config.get("_name_or_path")
        if isinstance(name_or_path, str) and name_or_path.strip():
            return name_or_path
        if "model_id" in config and isinstance(config["model_id"], str):
            return config["model_id"]
        return model_dir.name

    @staticmethod
    def _infer_hf_repo(config: dict) -> str | None:
        candidate = config.get("_name_or_path") or config.get("original_repo")
        if isinstance(candidate, str) and "/" in candidate:
            return candidate
        return None

    @staticmethod
    def _collect_metadata(config: dict) -> dict[str, str]:
        metadata: dict[str, str] = {}
        for key in (
            "model_type",
            "hidden_size",
            "num_hidden_layers",
            "architecture",
            "rope_scaling",
        ):
            value = config.get(key)
            if value is None:
                continue
            metadata[key] = (
                json.dumps(value) if isinstance(value, dict | list) else str(value)
            )

        quant_cfg = config.get("quantization_config") or config.get("quantization")
        if isinstance(quant_cfg, dict):
            bits = quant_cfg.get("bits") or quant_cfg.get("num_bits")
            if bits:
                metadata["quantization_bits"] = str(bits)
        return metadata

    @staticmethod
    def _casefold_alias(value: str) -> str:
        return re.sub(r"\s+", "", value.lower())

    def _build_aliases(
        self, canonical_id: str, dirname: str, hf_repo: str | None
    ) -> set[str]:
        aliases = {canonical_id, canonical_id.lower(), dirname, dirname.lower()}
        aliases.add(self._casefold_alias(canonical_id))
        aliases.add(self._casefold_alias(dirname))
        if hf_repo:
            aliases.add(hf_repo)
            aliases.add(hf_repo.lower())
            aliases.add(self._casefold_alias(hf_repo))
        return {alias for alias in aliases if alias}

    def _build_tokens(
        self,
        canonical_id: str,
        dirname: str,
        hf_repo: str | None,
        metadata: dict[str, str],
    ) -> set[str]:
        tokens: set[str] = set()
        for value in filter(None, [canonical_id, dirname, hf_repo]):
            tokens.update(self._tokenize(value))

        for value in metadata.values():
            tokens.update(self._tokenize(value))

        return tokens

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        raw = text.lower()
        fragments = set(re.split(r"[^a-z0-9]+", raw))
        fragments.discard("")
        collapsed = re.sub(r"[^a-z0-9]", "", raw)
        if collapsed:
            fragments.add(collapsed)
        # Capture shorthand like "8b" when present
        matches = re.findall(r"\d+\.?\d*b", raw)
        fragments.update(matches)
        return fragments
