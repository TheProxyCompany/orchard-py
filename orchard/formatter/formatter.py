import json
import logging
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader

from orchard.formatter.control_tokens import ControlTokens, load_control_tokens

logger = logging.getLogger(__name__)
_PROFILE_ROOT = Path(__file__).parent / "profiles"
_PROFILE_DIRS: dict[str, Path] | None = None


def _profile_dirs() -> dict[str, Path]:
    global _PROFILE_DIRS
    if _PROFILE_DIRS is None:
        profile_dirs = {}
        for profile_dir in _PROFILE_ROOT.iterdir():
            control_tokens_path = profile_dir / "control_tokens.json"
            if not control_tokens_path.is_file():
                continue
            with open(control_tokens_path) as f:
                control_tokens = json.load(f)
            profile_dirs[profile_dir.name] = profile_dir
            for model_type in control_tokens.get("model_types", []):
                if isinstance(model_type, str) and model_type:
                    profile_dirs[model_type] = profile_dir
        _PROFILE_DIRS = profile_dirs
    return _PROFILE_DIRS


def determine_model_type(config: dict) -> str:
    """Determine the architecture model type from model config."""
    model_type = config.get("model_type")
    if not isinstance(model_type, str) or not model_type:
        raise ValueError("Formatter config must include a non-empty 'model_type'")
    return model_type


def determine_template_type(config: dict) -> str:
    """Determine the Pantheon template type from model config."""
    template_type = config.get("template_type")
    if isinstance(template_type, str) and template_type:
        return template_type

    model_type = determine_model_type(config)
    model_name = ""
    for key in ("_name_or_path", "model_id", "original_repo"):
        value = config.get(key)
        if isinstance(value, str):
            model_name = value.lower()
            break
    if model_type == "phi3" and "phi-4-reasoning" in model_name:
        return "phi4_reasoning"

    if model_type == "llama" or model_type == "llama3":
        return "llama3"

    if model_type == "moondream3" or model_type == "moondream":
        return "moondream3"

    if model_type in ("qwen3_5", "qwen3_5_text", "qwen3_5_moe"):
        return "qwen3_5"

    if model_type in ("gemma4", "gemma4_text"):
        return "gemma4"

    return model_type


def find_shared_template_dir(profile_dir: Path) -> Path | None:
    direct_root = profile_dir.parent
    if direct_root and (direct_root / "tool_macros.jinja").is_file():
        return direct_root

    for ancestor in profile_dir.parents:
        candidate = ancestor / "Pantheon"
        if (candidate / "tool_macros.jinja").is_file():
            return candidate

    return None


# from https://github.com/huggingface/transformers/blob/7769f660935b5d48b73bf6711d0a78b6f8f98739/src/transformers/utils/chat_template_utils.py#L447C1-L451C1
def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    # We override the built-in tojson filter because Jinja's default filter escapes HTML characters
    # We also expose some options like custom indents and separators
    return json.dumps(
        x,
        ensure_ascii=ensure_ascii,
        indent=indent,
        separators=separators,
        sort_keys=sort_keys,
    )


class ChatFormatter:
    """
    Handles the application of chat templates to conversation histories.
    """

    @property
    def image_placeholder(self) -> str:
        """Resolve the image placeholder token from capabilities or control tokens."""
        vision = self.capabilities.get("vision", {})
        # Explicit placeholder (e.g. moondream: vision.placeholders.image)
        placeholder = vision.get("placeholders", {}).get("image")
        if placeholder:
            return placeholder
        placeholder = vision.get("tokens", {}).get("placeholder")
        if placeholder:
            return placeholder
        # Vision start token (e.g. gemma: vision.tokens.start)
        start = vision.get("tokens", {}).get("start")
        if start:
            return start
        # Legacy fallback to control_tokens
        if self.control_tokens.start_image_token:
            return self.control_tokens.start_image_token
        return "<|image|>"

    @property
    def should_clip_image_placeholder(self) -> bool:
        """Whether to clip the image placeholder from text segments.

        If the placeholder is a real token (has a start_image_token or vision.tokens.start),
        it stays in the text for tokenization. If it's a synthetic placeholder
        (like <|image|>), it gets clipped.
        """
        vision = self.capabilities.get("vision", {})
        has_placeholder = bool(
            vision.get("placeholders", {}).get("image")
            or vision.get("tokens", {}).get("placeholder")
        )
        if has_placeholder:
            return True
        has_vision_tokens = bool(vision.get("tokens", {}).get("start"))
        has_start_token = bool(self.control_tokens.start_image_token)
        return not (has_vision_tokens or has_start_token)

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        tokenizer_config_path = self.model_path / "config.json"
        if not tokenizer_config_path.exists():
            raise FileNotFoundError(
                f"tokenizer_config.json not found in {self.model_path}"
            )
        with open(tokenizer_config_path) as f:
            tokenizer_config = json.load(f)
        self._configure(tokenizer_config)

    @classmethod
    def from_config(
        cls, model_path: str, tokenizer_config: dict[str, Any]
    ) -> "ChatFormatter":
        formatter = cls.__new__(cls)
        formatter.model_path = Path(model_path)
        cls._configure(formatter, dict(tokenizer_config))
        return formatter

    def _configure(self, tokenizer_config: dict[str, Any]) -> None:
        self.tokenizer_config = tokenizer_config
        model_type = determine_model_type(tokenizer_config)
        template_type = determine_template_type(tokenizer_config)
        self.model_type = model_type
        self.template_type = template_type
        profile_dir = _PROFILE_ROOT / template_type
        if not profile_dir.is_dir():
            profile_dir = _profile_dirs().get(template_type, profile_dir)
        if not profile_dir.is_dir():
            raise ValueError(
                f"Profile directory for template_type '{template_type}' not found at "
                f"{profile_dir}"
            )
        self.profile_dir = profile_dir

        self.control_tokens: ControlTokens = load_control_tokens(profile_dir)

        caps_path = profile_dir / "capabilities.yaml"
        generation_path = profile_dir / "generation.yaml"
        if not caps_path.is_file():
            raise ValueError(
                f"Profile for template_type '{template_type}' is missing capabilities.yaml"
            )
        if not generation_path.is_file():
            raise ValueError(
                f"Profile for template_type '{template_type}' is missing generation.yaml"
            )
        self.capabilities = yaml.safe_load(caps_path.read_text()) or {}
        self.generation = yaml.safe_load(generation_path.read_text()) or {}

        # 3. Set up Jinja2 environment
        loader_paths = [str(profile_dir)]
        shared_template_dir = find_shared_template_dir(profile_dir)
        if shared_template_dir is not None:
            loader_paths.append(str(shared_template_dir))

        self.jinja_env = Environment(
            loader=FileSystemLoader(loader_paths), trim_blocks=True, lstrip_blocks=True
        )
        self.jinja_env.filters["tojson"] = tojson
        self.template = self.jinja_env.get_template("chat_template.jinja")

    def apply_template(
        self,
        conversation: list[dict[str, Any]],
        add_generation_prompt: bool = True,
        reasoning: bool = False,
        task: str | None = None,
        prefill: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        """
        Applies the loaded chat template to a conversation.

        Args:
            conversation: A list of message dictionaries, e.g., [{"role": "user", "content": "..."}].
            add_generation_prompt: Whether to add the assistant prompt turn.
            reasoning: Whether to add the conditional reasoning prompt logic.
            task: Optional task name for task-specific formatting (e.g., "caption_normal", "detect").
            prefill: Optional string to prefill the assistant response with.
            tools: Optional list of tool schema dicts to render inline.
        Returns:
            A single, fully formatted string ready for tokenization.
        """
        context = {
            "interactions": conversation,
            "messages": conversation,
            "add_generation_prompt": add_generation_prompt,
            "begin_of_text": self.control_tokens.begin_of_text,
            "end_of_sequence": self.control_tokens.end_of_sequence,
            "end_of_message": self.control_tokens.end_of_message,
            "start_image_token": self.control_tokens.start_image_token,
            "end_image_token": self.control_tokens.end_image_token,
            "reasoning": reasoning,
            "task": task,
            "roles": self.control_tokens.roles.model_dump(),
            "prefill": prefill,
            "capabilities": self.capabilities,
            "model_config": self.tokenizer_config,
            "tools": tools,
            "reasoning_effort": reasoning_effort or "medium",
        }
        return self.template.render(**context)

    def get_coord_placeholder(self) -> str | None:
        """Extract coord placeholder from capabilities.yaml (pointing or gaze_detection)."""
        for cap_name in ("pointing", "gaze_detection"):
            cap = self.capabilities.get(cap_name, {})
            placeholder = cap.get("placeholders", {}).get("coord")
            if placeholder:
                return placeholder
        return None

    def get_tool_calling_tokens(self) -> dict[str, Any]:
        """Extract tool calling delimiter tokens from capabilities.yaml."""
        tool_caps = self.capabilities.get("tool_calling", {})
        formats = tool_caps.get("formats") or []

        serialized_formats: list[dict[str, str]] = []
        section_start = ""
        section_end = ""

        for index, fmt in enumerate(formats):
            tokens = fmt.get("tokens", {})
            serialized_formats.append(
                {
                    "name": fmt.get("name", ""),
                    "call_start": tokens.get("start", ""),
                    "inline_start": tokens.get("inline_start", ""),
                    "channel": tokens.get("channel", ""),
                    "recipient_prefix": tokens.get("recipient_prefix", ""),
                    "constraint_prefix": tokens.get("constraint_prefix", ""),
                    "constraint": tokens.get("constraint", ""),
                    "message": tokens.get("message", ""),
                    "call_end": tokens.get("end", ""),
                }
            )
            if index == 0:
                section_start = tokens.get("section_start", "")
                section_end = tokens.get("section_end", "")

        return {
            "formats": serialized_formats,
            "section_start": section_start,
            "section_end": section_end,
        }

    def get_output_frame_tokens(self) -> dict[str, str]:
        framing = self.capabilities.get("output_framing", {})
        if not isinstance(framing, dict):
            return {}

        tokens: dict[str, str] = {}
        markers = framing.get("markers", {})
        if isinstance(markers, dict):
            tokens.update(
                {
                    f"marker.{name}": str(value)
                    for name, value in markers.items()
                    if value
                }
            )

        channels = framing.get("channels", {})
        if isinstance(channels, dict):
            tokens.update(
                {
                    f"channel.{name}": str(value)
                    for name, value in channels.items()
                    if value
                }
            )

        return tokens

    def get_thinking_tokens(self) -> dict[str, str]:
        """Extract generated-output thinking delimiters from capabilities.yaml."""
        tokens = self.capabilities.get("thinking", {}).get("tokens", {})
        return {
            "start": str(tokens.get("start", "")),
            "end": str(tokens.get("end", "")),
        }

    def get_generation_defaults(self, profile: str = "default") -> dict[str, Any]:
        """Return sampling defaults from generation.yaml."""
        value = self.generation.get(profile)
        if not isinstance(value, dict):
            value = self.generation.get("default")
        return dict(value) if isinstance(value, dict) else {}

    def supports_native_thinking(self) -> bool:
        return bool(self.capabilities.get("thinking", {}).get("native"))
