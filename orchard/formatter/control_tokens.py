import json
from pathlib import Path

from pydantic import BaseModel


class Role(BaseModel):
    role_name: str
    role_start_tag: str
    role_end_tag: str


class RoleTags(BaseModel):
    system: Role | None = None
    agent: Role | None = None
    user: Role | None = None
    tool: Role | None = None


class ControlTokens(BaseModel):
    """Structural control tokens for model templates.

    Turn delimiters and role tags only. Capability-specific tokens
    (tool calling, thinking, vision) live in capabilities.yaml.
    """

    template_type: str
    begin_of_text: str
    end_of_message: str
    end_of_sequence: str
    start_image_token: str | None = None
    end_image_token: str | None = None

    roles: RoleTags


def load_control_tokens(profile_dir: Path) -> ControlTokens:
    """Load control tokens from the given profile directory."""
    control_tokens_path = profile_dir / "control_tokens.json"
    if not control_tokens_path.is_file():
        raise FileNotFoundError(
            f"control_tokens.json not found in profile directory {profile_dir}"
        )
    with open(control_tokens_path) as f:
        data = json.load(f)
    return ControlTokens(**data)
