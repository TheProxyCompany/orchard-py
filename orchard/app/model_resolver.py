"""Model resolution utilities for mapping model identifiers to local or HuggingFace cached assets."""

from __future__ import annotations

import importlib.util
import json
import logging
import re
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

logger = logging.getLogger(__name__)

__all__ = [
    "ModelResolutionError",
    "ModelResolver",
    "ResolvedModel",
]

@dataclass(slots=True)
class ResolvedModel:
    """Result of resolving a model identifier to a local path."""

    canonical_id: str
    model_path: Path
    source: str  # "local" | "hf_cache" | "hf_hub"
    metadata: dict[str, str] = field(default_factory=dict)
    hf_repo: str | None = None


class ModelResolutionError(RuntimeError):
    """Raised when a model identifier cannot be resolved."""

    def __init__(self, message: str, candidates: Sequence[str] | None = None):
        super().__init__(message)
        self.candidates = list(candidates or [])


class IncompleteSnapshotError(RuntimeError):
    """Raised when a cached HuggingFace snapshot is missing required weights."""


class ModelResolver:
    """Resolves model identifiers to local filesystem paths.

    Resolution order:
    1. Local filesystem path (absolute or relative) -> use directly
    2. Treat as HF repo ID -> resolve via HuggingFace
    """

    def __init__(self):
        # Cache of resolved models to avoid repeated HF lookups
        self._resolved_cache: dict[str, ResolvedModel] = {}

    def resolve(self, requested_id: str) -> ResolvedModel:
        """Resolve a model identifier to a local filesystem path.

        Args:
            requested_id: Model identifier - can be:
                - Local path: /path/to/model or ./relative/path
                - HF repo ID: meta-llama/Llama-3.1-8B-Instruct

        Returns:
            ResolvedModel with the local path and metadata.

        Raises:
            ModelResolutionError: If the model cannot be resolved.
        """
        identifier = requested_id.strip()
        if not identifier:
            raise ModelResolutionError("Model identifier cannot be empty.")

        # Check cache first
        cache_key = identifier.lower()
        if cache_key in self._resolved_cache:
            return self._resolved_cache[cache_key]

        # 1. Direct filesystem path (absolute or relative)
        resolved = self._try_local_path(identifier)
        if resolved is not None:
            self._resolved_cache[cache_key] = resolved
            return resolved

        # 2. Treat as HF repo ID
        resolved = self._resolve_huggingface(identifier)
        self._resolved_cache[cache_key] = resolved
        return resolved

    def _try_local_path(self, identifier: str) -> ResolvedModel | None:
        """Try to resolve identifier as a local filesystem path."""
        path = Path(identifier)

        # Check absolute path
        if path.is_absolute():
            if path.exists() and path.is_dir():
                return self._build_resolved_model(path, source="local")
            return None

        # Check relative path (relative to CWD)
        if path.exists() and path.is_dir():
            return self._build_resolved_model(path.resolve(), source="local")

        return None

    def _resolve_huggingface(
        self, repo_id: str
    ) -> ResolvedModel:
        """Resolve a HuggingFace repo ID to a local cache path.

        Downloads the model if not already cached.
        """
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
            # Try local cache first
            path = Path(
                snapshot_download(
                    repo_id, local_files_only=True, allow_patterns=allow_patterns
                )
            )
            self._ensure_hf_weights_complete(path)
            source = "hf_cache"
        except (LocalEntryNotFoundError, IncompleteSnapshotError):
            # Download from HuggingFace Hub
            try:
                path = Path(
                    snapshot_download(
                        repo_id, local_files_only=False, allow_patterns=allow_patterns
                    )
                )
                self._ensure_hf_weights_complete(path)
                source = "hf_hub"
            except Exception as e:
                raise ModelResolutionError(
                    f"Failed to download model '{repo_id}' from HuggingFace: {e}"
                ) from e

        return self._build_resolved_model(
            path, source=source, canonical_id=repo_id, hf_repo=repo_id
        )

    def _build_resolved_model(
        self,
        model_path: Path,
        source: str,
        canonical_id: str | None = None,
        hf_repo: str | None = None,
    ) -> ResolvedModel:
        """Build a ResolvedModel from a model directory."""
        model_path = model_path.resolve()

        # Ensure model has all required files for inference
        config = self._ensure_model_complete(model_path)
        metadata = self._collect_metadata(config)

        # Determine canonical ID
        if canonical_id is None:
            canonical_id = self._determine_canonical_id(config, model_path)

        # Infer HF repo from config if not provided
        if hf_repo is None:
            hf_repo = self._infer_hf_repo(config)

        return ResolvedModel(
            canonical_id=canonical_id,
            model_path=model_path,
            source=source,
            metadata=metadata,
            hf_repo=hf_repo,
        )

    # -------------------------------------------------------------------------
    # Model Completeness
    # -------------------------------------------------------------------------

    @staticmethod
    def _ensure_hf_weights_complete(model_dir: Path) -> None:
        index_file = model_dir / "model.safetensors.index.json"
        if index_file.exists():
            with index_file.open("r", encoding="utf-8") as f:
                index_payload = json.load(f)
            weight_map = index_payload.get("weight_map", {})
            shard_names = sorted({name for name in weight_map.values() if isinstance(name, str)})
            if not shard_names:
                raise IncompleteSnapshotError(
                    f"Cached HuggingFace snapshot '{model_dir}' has no declared weight shards."
                )
            missing_shards = [
                shard_name
                for shard_name in shard_names
                if not ModelResolver._is_materialized_weight_file(model_dir / shard_name)
            ]
            if missing_shards:
                raise IncompleteSnapshotError(
                    f"Cached HuggingFace snapshot '{model_dir}' is missing weight shard(s): "
                    + ", ".join(missing_shards)
                )
            return

        single_file = model_dir / "model.safetensors"
        if single_file.exists() or single_file.is_symlink():
            if not ModelResolver._is_materialized_weight_file(single_file):
                raise IncompleteSnapshotError(
                    f"Cached HuggingFace snapshot '{model_dir}' has an incomplete model.safetensors."
                )
            return

        raise IncompleteSnapshotError(
            f"Cached HuggingFace snapshot '{model_dir}' has no materialized safetensors weights."
        )

    @staticmethod
    def _is_materialized_weight_file(path: Path) -> bool:
        candidate = path
        if candidate.is_symlink():
            try:
                candidate = candidate.resolve(strict=True)
            except OSError:
                return False

        if not candidate.is_file() or candidate.name.endswith(".incomplete"):
            return False
        return candidate.stat().st_size > 0

    def _ensure_model_complete(self, model_dir: Path) -> dict:
        """Ensure model directory has all required files for inference.

        HuggingFace models may:
        - Define config in Python (auto_map pattern)
        - Reference tokenizers from other repos
        - Have other external dependencies

        This method ensures all required assets are present and returns the config.
        """
        config = self._load_config(model_dir)

        # Config completeness: augment from Python if using auto_map
        if "auto_map" in config:
            self._ensure_config_complete(model_dir, config)

        # Tokenizer completeness: resolve external tokenizer if missing
        if not self._has_tokenizer(model_dir):
            self._ensure_tokenizer_complete(model_dir)

        return config

    @staticmethod
    def _load_config(model_dir: Path) -> dict:
        """Load config.json from a model directory."""
        config_file = model_dir / "config.json"
        if not config_file.exists():
            raise ModelResolutionError(
                f"Model directory '{model_dir}' is missing config.json."
            )
        with config_file.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _ensure_config_complete(self, model_dir: Path, config: dict) -> None:
        """Augment config.json with Python-defined config if needed."""
        config_py = model_dir / "config.py"
        if not config_py.exists():
            return

        python_config = self._load_python_config(model_dir, config)
        if not python_config:
            return

        # Only merge model architecture keys (text, vision, region), not tokenizer
        # PIE expects the architecture config but handles tokenizer separately
        architecture_keys = ["text", "vision", "region"]
        new_keys = [
            k for k in architecture_keys if k in python_config and k not in config
        ]

        if not new_keys:
            # Still check if dtype needs to be added
            if "dtype" not in config and "torch_dtype" in config:
                config["dtype"] = config["torch_dtype"]
                new_keys = ["dtype"]
            else:
                return
        else:
            # Merge Python config into JSON config
            for key in new_keys:
                config[key] = python_config[key]

            # Ensure dtype is set (PIE expects this field)
            if "dtype" not in config and "torch_dtype" in config:
                config["dtype"] = config["torch_dtype"]
                new_keys.append("dtype")

        # Write augmented config back to disk for PIE to read
        config_file = model_dir / "config.json"
        try:
            with config_file.open("w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            logger.info(f"Augmented config.json for {model_dir.name}: added {new_keys}")
        except OSError as e:
            logger.warning(f"Failed to write augmented config: {e}")

    @staticmethod
    def _has_tokenizer(model_dir: Path) -> bool:
        """Check if model directory has tokenizer files."""
        return (model_dir / "tokenizer.json").exists() or (
            model_dir / "tokenizer.model"
        ).exists()

    def _ensure_tokenizer_complete(self, model_dir: Path) -> None:
        """Find and fetch tokenizer from external source if referenced."""
        tokenizer_repo = self._find_tokenizer_reference(model_dir)
        if not tokenizer_repo:
            logger.warning(
                f"No tokenizer found for {model_dir.name} and no external reference detected"
            )
            return

        logger.info(f"Fetching tokenizer from {tokenizer_repo} for {model_dir.name}")
        self._fetch_tokenizer(tokenizer_repo, model_dir)

    @staticmethod
    def _find_tokenizer_reference(model_dir: Path) -> str | None:
        """Scan Python files for Tokenizer.from_pretrained() calls."""
        pattern = r'Tokenizer\.from_pretrained\(["\']([^"\']+)["\']\)'

        for py_file in model_dir.glob("*.py"):
            try:
                content = py_file.read_text(encoding="utf-8")
                if match := re.search(pattern, content):
                    return match.group(1)
            except OSError:
                continue

        return None

    def _fetch_tokenizer(self, repo_id: str, target_dir: Path) -> None:
        """Download tokenizer files from a HuggingFace repo and copy to target."""
        tokenizer_patterns = [
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ]

        try:
            tokenizer_path = Path(
                snapshot_download(
                    repo_id,
                    allow_patterns=tokenizer_patterns,
                    local_files_only=False,
                )
            )

            # Copy tokenizer files to target directory
            for pattern in tokenizer_patterns:
                src = tokenizer_path / pattern
                if src.exists():
                    dst = target_dir / pattern
                    if not dst.exists():
                        shutil.copy2(src, dst)
                        logger.debug(f"Copied {pattern} to {target_dir.name}")

            logger.info(f"Tokenizer files copied from {repo_id} to {target_dir.name}")

        except Exception as e:
            logger.warning(f"Failed to fetch tokenizer from {repo_id}: {e}")

    @staticmethod
    def _load_python_config(model_dir: Path, json_config: dict) -> dict | None:
        """Load config from a Python config.py file.

        This handles the HuggingFace auto_map pattern where model config
        is defined in Python dataclasses rather than JSON.
        """
        config_py = model_dir / "config.py"
        if not config_py.exists():
            return None

        try:
            # Temporarily add model_dir to sys.path to import config.py
            sys.path.insert(0, str(model_dir))
            try:
                # Load config.py as a module
                spec = importlib.util.spec_from_file_location(
                    "_model_config", config_py
                )
                if spec is None or spec.loader is None:
                    return None

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Try to find a config class based on model_type
                config_class = ModelResolver._find_config_class(module, json_config)
                if config_class is None:
                    return None

                # Instantiate with defaults and convert to dict
                config_instance = config_class()
                return ModelResolver._dataclass_to_dict(config_instance)

            finally:
                # Clean up sys.path
                if str(model_dir) in sys.path:
                    sys.path.remove(str(model_dir))

        except Exception as e:
            logger.warning(f"Failed to load Python config from {config_py}: {e}")
            return None

    @staticmethod
    def _find_config_class(module: Any, json_config: dict) -> type | None:
        """Find the appropriate config class in a config.py module."""
        model_type = json_config.get("model_type", "")

        # Try common config class naming patterns
        candidates = [
            f"{model_type.title().replace('-', '').replace('_', '')}Config",  # e.g., Moondream3Config
            f"{model_type.title()}Config",  # e.g., MoondreamConfig
            "Config",
            "ModelConfig",
        ]

        for name in candidates:
            if hasattr(module, name):
                cls = getattr(module, name)
                if isinstance(cls, type):
                    return cls

        # Fallback: look for any class ending in "Config"
        for name in dir(module):
            if name.endswith("Config") and not name.startswith("_"):
                cls = getattr(module, name)
                if isinstance(cls, type):
                    return cls

        return None

    @staticmethod
    def _dataclass_to_dict(obj: Any) -> dict:
        """Recursively convert a dataclass instance to a dict."""
        if hasattr(obj, "__dataclass_fields__"):
            return {
                k: ModelResolver._dataclass_to_dict(v) for k, v in obj.__dict__.items()
            }
        elif isinstance(obj, dict):
            return {k: ModelResolver._dataclass_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list | tuple):
            return [ModelResolver._dataclass_to_dict(v) for v in obj]
        else:
            return obj

    @staticmethod
    def _determine_canonical_id(config: dict, model_dir: Path) -> str:
        """Determine the canonical model ID from config or directory name."""
        name_or_path = config.get("_name_or_path")
        if isinstance(name_or_path, str) and name_or_path.strip():
            return name_or_path
        if "model_id" in config and isinstance(config["model_id"], str):
            return config["model_id"]
        return model_dir.name

    @staticmethod
    def _infer_hf_repo(config: dict) -> str | None:
        """Infer HuggingFace repo ID from config."""
        candidate = config.get("_name_or_path") or config.get("original_repo")
        if isinstance(candidate, str) and "/" in candidate:
            return candidate
        return None

    @staticmethod
    def _collect_metadata(config: dict) -> dict[str, str]:
        """Extract relevant metadata from model config."""
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
