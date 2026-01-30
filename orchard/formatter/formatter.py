import json
import logging
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from orchard.formatter.control_tokens import ControlTokens, load_control_tokens

logger = logging.getLogger(__name__)


def determine_model_type(config: dict) -> str:
    """Determine the model type from the model path."""
    model_type = config.get("model_type", "llama")
    if model_type == "llama" or model_type == "llama3":
        return "llama3"

    if model_type == "moondream3" or model_type == "moondream":
        return "moondream3"

    return model_type


class ChatFormatter:
    """
    Handles the application of chat templates to conversation histories.
    """

    @property
    def should_clip_image_placeholder(self) -> bool:
        """Whether to clip the image placeholder. If the model does not have a start image token, we need to clip the image placeholder."""
        has_start_token = (
            self.control_tokens.start_image_token is not None
            and self.control_tokens.start_image_token != ""
        )
        return not has_start_token

    @property
    def default_image_placeholder(self) -> str:
        """The default image placeholder to use if the model does not have a start image token."""
        return "<|image|>"

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)

        # 1. Load tokenizer_config to determine model family
        tokenizer_config_path = self.model_path / "config.json"
        if not tokenizer_config_path.exists():
            raise FileNotFoundError(
                f"tokenizer_config.json not found in {self.model_path}"
            )
        with open(tokenizer_config_path) as f:
            self.tokenizer_config = json.load(f)

        model_type = determine_model_type(self.tokenizer_config)
        profile_dir = Path(__file__).parent / "profiles" / model_type
        if not profile_dir.is_dir():
            raise ValueError(
                f"Profile directory for model_type '{model_type}' not found at "
                f"{profile_dir}"
            )
        self.profile_dir = profile_dir

        self.control_tokens: ControlTokens = load_control_tokens(profile_dir)

        # 3. Set up Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(profile_dir), trim_blocks=True, lstrip_blocks=True
        )
        self.template = self.jinja_env.get_template("chat_template.jinja")

    def apply_template(
        self,
        conversation: list[dict[str, Any]],
        add_generation_prompt: bool = True,
        reasoning: bool = False,
        task: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """
        Applies the loaded chat template to a conversation.

        Args:
            conversation: A list of message dictionaries, e.g., [{"role": "user", "content": "..."}].
            add_generation_prompt: Whether to add the assistant prompt turn.
            reasoning: Whether to add the conditional reasoning prompt logic.
            task: Optional task name for task-specific formatting (e.g., "caption_normal", "detect").
            tools: Optional list of tool/function schemas to include in the prompt.
        Returns:
            A single, fully formatted string ready for tokenization.
        """
        context = {
            "interactions": conversation,
            "add_generation_prompt": add_generation_prompt,
            "begin_of_text": self.control_tokens.begin_of_text,
            "end_of_sequence": self.control_tokens.end_of_sequence,
            "end_of_message": self.control_tokens.end_of_message,
            "start_image_token": self.control_tokens.start_image_token,
            "end_image_token": self.control_tokens.end_image_token,
            "thinking_start_token": self.control_tokens.thinking_start_token,
            "thinking_end_token": self.control_tokens.thinking_end_token,
            "reasoning": reasoning,
            "task": task,
            "roles": self.control_tokens.roles.model_dump(),
            "tools": tools,
        }
        return self.template.render(**context)
