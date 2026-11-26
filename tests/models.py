"""Models under test, keyed by architecture.

What the suites certify is the *architecture* in the engine, the Pantheon
template_type, not a vendor's checkpoint name. One row per architecture;
use the smallest checkpoint that exercises it. Add a model by adding a row.
"""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Model:
    template_type: str  # Pantheon profile / architecture under test
    checkpoint: str  # smallest HF checkpoint that exercises it
    thinking: bool | str  # native thinking; "required" means it cannot be disabled
    vision: bool  # native vision
    tools: bool  # native tool calling


_ALL_MODELS = [
    Model(
        "llama3",
        "meta-llama/Llama-3.1-8B-Instruct",
        thinking=False,
        vision=False,
        tools=True,
    ),
    Model("gemma4", "google/gemma-4-E2B-it", thinking=True, vision=True, tools=True),
    Model("qwen3_5", "Qwen/Qwen3.5-4B", thinking=True, vision=False, tools=True),
    Model(
        "moondream3",
        "moondream/moondream3-preview",
        thinking=True,
        vision=True,
        tools=False,
    ),
    Model(
        "afmoe",
        "mlx-community/Trinity-Mini-4bit",
        thinking=True,
        vision=False,
        tools=True,
    ),
    Model(
        "lfm2_5",
        "LiquidAI/LFM2.5-8B-A1B",
        thinking="required",
        vision=False,
        tools=True,
    ),
    Model(
        "olmo_hybrid",
        "allenai/Olmo-Hybrid-Instruct-DPO-7B",
        thinking=False,
        vision=False,
        tools=True,
    ),
    Model(
        "nemotron_h",
        "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        thinking=True,
        vision=False,
        tools=True,
    ),
    Model(
        "granite_switch",
        "mlx-community/granite-4.1-30b-4bit",
        thinking=False,
        vision=False,
        tools=True,
    ),
    Model(
        "gpt_oss", "openai/gpt-oss-20b", thinking=True, vision=False, tools=True
    ),  # MXFP4 expert weights
]

_IMAGE_PIPELINE_MODELS = [
    "ideogram-ai/ideogram-4-fp8",
    "black-forest-labs/FLUX.2-klein-4B",
    "Qwen/Qwen-Image-Edit",
]

_AUDIO_PIPELINE_MODELS = [
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "mlx-community/parakeet-tdt-0.6b-v3",
    "Qwen/Qwen3-ASR-0.6B",
    "Qwen/Qwen3-ASR-1.7B",
]


def _select_models() -> list[Model]:
    raw = os.getenv("ORCHARD_TEST_MODELS", "").strip()
    if not raw:
        return list(_ALL_MODELS)
    requested = [token.strip() for token in raw.split(",") if token.strip()]
    selected = [
        model
        for model in _ALL_MODELS
        if model.template_type in requested or model.checkpoint in requested
    ]
    matched = {model.template_type for model in selected} | {
        model.checkpoint for model in selected
    }
    unknown = [token for token in requested if token not in matched]
    if unknown or not selected:
        raise ValueError(
            f"Unsupported ORCHARD_TEST_MODELS={raw!r}; unknown={unknown!r}"
        )
    return selected


def _select_pipeline_models() -> tuple[str | None, list[str]]:
    family = os.getenv("ORCHARD_TEST_PIPELINE_FAMILY", "").strip().lower()
    if not family:
        return None, _IMAGE_PIPELINE_MODELS + _AUDIO_PIPELINE_MODELS
    if family == "audio":
        return "audio", list(_AUDIO_PIPELINE_MODELS)
    if family == "image":
        return "image", list(_IMAGE_PIPELINE_MODELS)
    if family == "none":
        return "none", []
    raise ValueError(
        f"Unsupported ORCHARD_TEST_PIPELINE_FAMILY={family!r}; expected audio, image, or none"
    )


MODELS = _select_models()

# Modal tool models the pipeline suite activates. Hydrate them one at a
# time on top of the resident chat matrix: concurrent hydration spikes
# wired memory past the Metal limit (silent engine abort / failed requests).
PIPELINE_FAMILY, PIPELINE_TOOL_MODELS = _select_pipeline_models()
