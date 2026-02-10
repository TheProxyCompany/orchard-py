from __future__ import annotations

import base64
import json
import logging
import re
import struct
from binascii import Error as BinasciiError
from collections.abc import Iterable
from typing import Any

from orchard.formatter import ChatFormatter

logger = logging.getLogger(__name__)


class CapabilityInput:
    """Represents a capability input (coord, size) with name and payload bytes."""

    __slots__ = ("name", "payload")

    def __init__(self, name: str, payload: bytes) -> None:
        self.name = name
        self.payload = payload


DATA_URL_BASE64_PATTERN = re.compile(
    r"^data:(?P<mime>[\w\-/+.]+);base64,(?P<data>[A-Za-z0-9+/=]+)$"
)


class _RenderableText:
    """Wrapper that renders as text but exposes an indexable `type` field for Jinja."""

    __slots__ = ("_text",)
    _TYPE = "text"

    def __init__(self, text: str) -> None:
        self._text = text

    def __getitem__(self, key: str) -> str:
        if key == "type":
            return self._TYPE
        raise KeyError(key)

    def __str__(self) -> str:
        return self._text


class _RenderableImage:
    """Placeholder wrapper that renders as empty text and reports `type=image`."""

    __slots__ = ()
    _TYPE = "image"

    def __getitem__(self, key: str) -> str:
        if key == "type":
            return self._TYPE
        raise KeyError(key)

    def __str__(self) -> str:
        return ""


class _RenderableCapability:
    """Placeholder wrapper for capability inputs (coord, size). Renders as empty."""

    __slots__ = ()
    _TYPE = "capability"

    def __getitem__(self, key: str) -> str:
        if key == "type":
            return self._TYPE
        raise KeyError(key)

    def __str__(self) -> str:
        return ""


def _decode_image_payload(data_url: str) -> bytes:
    match_data = DATA_URL_BASE64_PATTERN.match(data_url)
    if not match_data:
        raise ValueError("Invalid image data URL format.")
    base64_data = match_data.group("data")
    try:
        return base64.b64decode(base64_data, validate=True)
    except (BinasciiError, ValueError) as exc:
        raise ValueError("Invalid base64-encoded image content.") from exc


def _normalize_role(raw_role: str | None, available_roles: set[str]) -> str:
    if not raw_role:
        return "user"
    role_lower = raw_role.lower()
    alias_map = {
        "assistant": "agent",
        "model": "agent",
        "developer": "system",
    }
    normalized = alias_map.get(role_lower, role_lower)
    if normalized not in available_roles:
        logger.debug(
            "Role '%s' not found in formatter profile; using as-is.", normalized
        )
    return normalized


def _get_field(candidate: Any, key: str, default: Any = None) -> Any:
    """Retrieve a value from an object or mapping."""
    if isinstance(candidate, dict):
        return candidate.get(key, default)
    return getattr(candidate, key, default)


def _parse_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    """Parse tool calls so that function.arguments is a dict, not a JSON string."""
    parsed = []
    for tc in tool_calls:
        tc_dict = tc if isinstance(tc, dict) else tc.model_dump()
        fn = tc_dict.get("function", {})
        args = fn.get("arguments", "")
        if isinstance(args, str) and args:
            fn["arguments"] = json.loads(args)
        parsed.append(tc_dict)
    return parsed


def build_multimodal_messages(
    formatter: ChatFormatter,
    items: Iterable[Any],
    instructions: str | None = None,
) -> tuple[
    list[dict[str, Any]], list[bytes], list[CapabilityInput], list[tuple[str, int]]
]:
    """Build multimodal messages for template rendering.

    Returns:
        Tuple of (messages, image_buffers, capabilities, content_order).
        content_order is a list of (type, index) tuples indicating the order of
        multimodal content parts (e.g., [("image", 0), ("text", 0)]).
    """
    roles_model = formatter.control_tokens.roles.model_dump()
    available_roles = {name for name, value in roles_model.items() if value}

    messages: list[dict[str, Any]] = []
    image_buffers: list[bytes] = []
    capabilities: list[CapabilityInput] = []
    content_order: list[tuple[str, int]] = []

    if instructions:
        system_role = (
            "system"
            if "system" in available_roles
            else _normalize_role("system", available_roles)
        )
        messages.append({"role": system_role, "content": instructions})

    for message_index, message in enumerate(items):
        role = _normalize_role(_get_field(message, "role"), available_roles)
        content = _get_field(message, "content")
        raw_tool_calls = _get_field(message, "tool_calls")
        tool_calls = _parse_tool_calls(raw_tool_calls) if raw_tool_calls else None

        if isinstance(content, str):
            msg: dict[str, Any] = {"role": role, "content": content}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            messages.append(msg)
            continue

        if not isinstance(content, (list | tuple)):
            raise TypeError(
                "Message content must be a string or list of content parts."
            )

        parts: list[_RenderableText | _RenderableImage | _RenderableCapability] = []
        for part_index, content_part in enumerate(content):
            part_type = _get_field(content_part, "type")
            if not isinstance(part_type, str):
                raise TypeError(
                    f"Content part {part_index} in message {message_index} is missing a valid 'type'."
                )

            normalized_type = part_type.lower()
            if normalized_type in {"input_text", "text"}:
                text_value = _get_field(content_part, "text")
                if text_value is None:
                    raise ValueError(
                        f"Text content missing for part {part_index} in message {message_index}."
                    )
                parts.append(_RenderableText(str(text_value)))
            elif normalized_type in {"input_image", "image", "image_url"}:
                image_url = _get_field(content_part, "image_url")
                if isinstance(image_url, dict):
                    image_url = image_url.get("url") or image_url.get("data")
                if not isinstance(image_url, str):
                    raise TypeError(
                        f"Image content part {part_index} in message {message_index} missing image_url."
                    )
                decoded_bytes = _decode_image_payload(image_url)
                logger.info("Decoded image bytes: %d", len(decoded_bytes))
                content_order.append(("image", len(image_buffers)))
                image_buffers.append(decoded_bytes)
                parts.append(_RenderableImage())
            elif normalized_type == "capability":
                name = _get_field(content_part, "name")
                data = _get_field(content_part, "data")
                if not name or not isinstance(name, str):
                    raise ValueError(
                        f"Capability part {part_index} in message {message_index} missing 'name'."
                    )
                if not data or not isinstance(data, list | tuple):
                    raise ValueError(
                        f"Capability part {part_index} in message {message_index} missing 'data' array."
                    )
                payload = struct.pack("<" + "f" * len(data), *data)
                content_order.append(("capability", len(capabilities)))
                capabilities.append(CapabilityInput(name, payload))
                parts.append(_RenderableCapability())
            else:
                logger.error(
                    "Unsupported content type in part %d of message %d: %s",
                    part_index,
                    message_index,
                    content_part,
                )
                raise ValueError(f"Unsupported content type: {part_type}")

        msg = {"role": role, "content": parts}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        messages.append(msg)

    return messages, image_buffers, capabilities, content_order


DEFAULT_COORD_PLACEHOLDER = "<|coord|>"


def build_multimodal_layout(
    prompt_text: str,
    image_buffers: list[bytes],
    capabilities: list[CapabilityInput],
    content_order: list[tuple[str, int]],
    placeholder_token: str,
    exclude_image_placeholder: bool,
    coord_placeholder: str | None = None,
) -> list[dict[str, Any]]:
    """Build the multimodal layout with images and capabilities at correct positions.

    Args:
        prompt_text: The rendered prompt text with image placeholders.
        image_buffers: List of image byte buffers.
        capabilities: List of capability inputs.
        content_order: Order of multimodal content parts from build_multimodal_messages.
        placeholder_token: The image placeholder token (e.g., "<|image|>").
        exclude_image_placeholder: Whether to exclude the placeholder from text segments.
        coord_placeholder: Optional coordinate placeholder token (e.g., "<|coord|>").
            If provided and found in prompt_text, capabilities will be placed at
            placeholder positions instead of using content_order.

    Returns:
        List of layout segment dicts with type and length.
    """
    layout: list[dict[str, Any]] = []

    if not image_buffers and not capabilities:
        # Text-only case
        text_bytes = prompt_text.encode("utf-8")
        if not text_bytes:
            raise ValueError(
                "Response request must include at least one content segment."
            )
        layout.append({"type": "text", "length": len(text_bytes)})
        return layout

    # Find image placeholder positions
    image_matches = (
        list(re.finditer(re.escape(placeholder_token), prompt_text))
        if image_buffers
        else []
    )
    if len(image_matches) != len(image_buffers):
        logger.error(
            "Mismatch between rendered image placeholders (%d) and supplied images (%d).",
            len(image_matches),
            len(image_buffers),
        )
        raise ValueError(
            "Mismatch between image placeholders and supplied image parts."
        )

    # Find coord placeholder positions if coord_placeholder is provided
    coord_placeholder_token = coord_placeholder or DEFAULT_COORD_PLACEHOLDER
    coord_matches = list(re.finditer(re.escape(coord_placeholder_token), prompt_text))
    use_coord_placeholders = len(coord_matches) > 0

    if use_coord_placeholders:
        # Validate: coord placeholders should match coord capabilities
        coord_capabilities = [c for c in capabilities if c.name == "coord"]
        if len(coord_matches) != len(coord_capabilities):
            logger.error(
                "Mismatch between coord placeholders (%d) and coord capabilities (%d).",
                len(coord_matches),
                len(coord_capabilities),
            )
            raise ValueError(
                "Mismatch between coord placeholders and coord capability parts."
            )

        # Build combined list of all placeholders with their positions
        # Each entry: (position, type, index) where type is "image" or "coord"
        all_placeholders: list[
            tuple[int, int, str, int]
        ] = []  # (start, end, type, idx)

        for idx, match in enumerate(image_matches):
            all_placeholders.append((match.start(), match.end(), "image", idx))

        for idx, match in enumerate(coord_matches):
            all_placeholders.append((match.start(), match.end(), "coord", idx))

        # Sort by position
        all_placeholders.sort(key=lambda x: x[0])

        # Build layout by processing placeholders in order
        cursor = 0
        coord_cap_idx = 0

        for start, end, ptype, idx in all_placeholders:
            # Add text before this placeholder
            if ptype == "image":
                text_segment = prompt_text[
                    cursor : start if exclude_image_placeholder else end
                ]
            else:  # coord placeholder - always exclude from text
                text_segment = prompt_text[cursor:start]

            segment_bytes = text_segment.encode("utf-8")
            if segment_bytes:
                layout.append({"type": "text", "length": len(segment_bytes)})

            # Add the placeholder content
            if ptype == "image":
                layout.append({"type": "image", "length": len(image_buffers[idx])})
            else:  # coord
                cap = coord_capabilities[coord_cap_idx]
                layout.append(
                    {"type": "capability", "name": cap.name, "length": len(cap.payload)}
                )
                coord_cap_idx += 1

            cursor = end

        # Add remaining text after last placeholder
        tail_segment = prompt_text[cursor:]
        if tail_segment:
            tail_bytes = tail_segment.encode("utf-8")
            layout.append({"type": "text", "length": len(tail_bytes)})
    else:
        # Original behavior: Build layout following content_order
        # Images are at placeholder positions; capabilities go right after the preceding image
        cursor = 0
        image_idx = 0
        cap_idx = 0

        for content_type, _ in content_order:
            if content_type == "image":
                # Add text before this image
                match = image_matches[image_idx]
                text_segment = prompt_text[
                    cursor : match.start() if exclude_image_placeholder else match.end()
                ]
                segment_bytes = text_segment.encode("utf-8")
                if segment_bytes:
                    layout.append({"type": "text", "length": len(segment_bytes)})
                # Add the image
                layout.append(
                    {"type": "image", "length": len(image_buffers[image_idx])}
                )
                cursor = match.end()
                image_idx += 1
            elif content_type == "capability":
                # Add capability segment (capabilities don't consume text)
                cap = capabilities[cap_idx]
                layout.append(
                    {"type": "capability", "name": cap.name, "length": len(cap.payload)}
                )
                cap_idx += 1

        # Add remaining text after last image/capability
        tail_segment = prompt_text[cursor:]
        if tail_segment:
            tail_bytes = tail_segment.encode("utf-8")
            layout.append({"type": "text", "length": len(tail_bytes)})

    if not layout:
        raise ValueError("Response request must include at least one content segment.")

    return layout
