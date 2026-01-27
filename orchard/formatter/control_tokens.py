import json
from pathlib import Path

from pydantic import BaseModel, Field


class Role(BaseModel):
    role_name: str
    role_start_tag: str
    role_end_tag: str


class RoleTags(BaseModel):
    system: Role | None = None
    agent: Role | None = None
    user: Role | None = None
    tool: Role | None = None


class ToolCallingTokens(BaseModel):
    """Tokens for structured tool/function calling output."""

    call_start: str = ""
    call_end: str = ""
    section_start: str = ""
    section_end: str = ""
    name_separator: str = ""


class ControlTokens(BaseModel):
    """Control tokens for different model templates.

    This class defines the structure and access methods for control tokens used in
    various LLM template formats.
    """

    template_type: str
    begin_of_text: str
    end_of_message: str
    end_of_sequence: str
    start_image_token: str | None = None
    end_image_token: str | None = None
    thinking_start_token: str | None = None
    thinking_end_token: str | None = None
    capabilities: dict[str, str] = Field(default_factory=dict)
    tool_calling: ToolCallingTokens = Field(default_factory=ToolCallingTokens)

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
