"""Models under test, keyed by architecture.

What the suites certify is the *architecture* in the engine — the Pantheon
template_type — not a vendor's checkpoint name. One row per architecture;
use the smallest checkpoint that exercises it. Add a model by adding a row.

``ORCHARD_TEST_MODELS`` (comma-separated template types, e.g. ``gpt_oss`` or
``gemma4,moondream3``) narrows the matrix for one session: ``MODELS`` holds
only the named rows, so the engine fixture preloads only those checkpoints and
every matrix test parametrizes over only them. Unset means the whole matrix.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    template_type: str  # Pantheon profile / architecture under test
    checkpoint: str  # smallest HF checkpoint that exercises it
    thinking: bool | str  # native thinking; "required" means it cannot be disabled
    vision: bool  # native vision
    tools: bool  # native tool calling


MATRIX = [
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


def select_models(spec: str | None, matrix: list[Model] = MATRIX) -> list[Model]:
    """The rows of ``matrix`` named in ``spec``, in matrix order.

    ``spec`` is a comma-separated list of template types; None or blank keeps
    the whole matrix. An unknown name raises instead of silently selecting
    nothing, so a typo fails at collection, before any engine starts.
    """
    names = [name.strip() for name in (spec or "").split(",") if name.strip()]
    if not names:
        return list(matrix)
    known = {model.template_type for model in matrix}
    unknown = sorted(set(names) - known)
    if unknown:
        raise ValueError(
            f"ORCHARD_TEST_MODELS names unknown template type(s) {unknown}; "
            f"known: {sorted(known)}"
        )
    return [model for model in matrix if model.template_type in names]


MODELS = select_models(os.environ.get("ORCHARD_TEST_MODELS"))

# Modal tool models the pipeline suite activates. Hydrate them one at a
# time on top of the resident chat matrix: concurrent hydration spikes
# wired memory past the Metal limit (silent engine abort / failed requests).
PIPELINE_TOOL_MODELS = [
    "ideogram-ai/ideogram-4-fp8",
    "black-forest-labs/FLUX.2-klein-4B",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "mlx-community/parakeet-tdt-0.6b-v3",
    "Qwen/Qwen3-ASR-0.6B",
    "Qwen/Qwen3-ASR-1.7B",
]
