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
        if key == "text":
            return self._text
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


class _RenderableAudio:
    """Placeholder wrapper that renders as empty text and reports `type=audio`."""

    __slots__ = ()
    _TYPE = "audio"

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
    list[dict[str, Any]],
    list[bytes],
    list[bytes],
    list[CapabilityInput],
    list[tuple[str, int]],
]:
    """Build multimodal messages for template rendering.

    Returns:
        Tuple of (messages, image_buffers, audio_buffers, capabilities, content_order).
        content_order is a list of (type, index) tuples indicating the order of
        multimodal content parts (e.g., [("image", 0), ("text", 0)]).
    """
    roles_model = formatter.control_tokens.roles.model_dump()
    available_roles = {name for name, value in roles_model.items() if value}

    messages: list[dict[str, Any]] = []
    image_buffers: list[bytes] = []
    audio_buffers: list[bytes] = []
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
        if isinstance(message, dict):
            msg = dict(message)
        elif hasattr(message, "model_dump"):
            msg = message.model_dump()
        else:
            msg = {"role": role, "content": content}

        if isinstance(content, str):
            msg["role"] = role
            msg["content"] = content
            if tool_calls is not None:
                msg["tool_calls"] = tool_calls
            messages.append(msg)
            continue

        if not isinstance(content, (list | tuple)):
            raise TypeError(
                "Message content must be a string or list of content parts."
            )

        parts: list[
            _RenderableText | _RenderableImage | _RenderableAudio | _RenderableCapability
        ] = []
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
                logger.debug("Decoded image bytes: %d", len(decoded_bytes))
                content_order.append(("image", len(image_buffers)))
                image_buffers.append(decoded_bytes)
                parts.append(_RenderableImage())
            elif normalized_type in {"input_audio", "audio"}:
                data = _get_field(content_part, "data")
                if isinstance(data, dict):
                    data = data.get("samples") or data.get("data")
                if not data or not isinstance(data, list | tuple):
                    raise ValueError(
                        f"Audio content part {part_index} in message {message_index} missing float32 'data' array."
                    )
                payload = struct.pack("<" + "f" * len(data), *data)
                content_order.append(("audio", len(audio_buffers)))
                audio_buffers.append(payload)
                parts.append(_RenderableAudio())
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

        msg["role"] = role
        msg["content"] = parts
        if tool_calls is not None:
            msg["tool_calls"] = tool_calls
        messages.append(msg)

    return messages, image_buffers, audio_buffers, capabilities, content_order


DEFAULT_COORD_PLACEHOLDER = "<|coord|>"


def build_multimodal_layout(
    prompt_text: str,
    image_buffers: list[bytes],
    audio_buffers: list[bytes],
    capabilities: list[CapabilityInput],
    content_order: list[tuple[str, int]],
    placeholder_token: str,
    exclude_image_placeholder: bool,
    audio_placeholder: str | None = None,
    coord_placeholder: str | None = None,
) -> list[dict[str, Any]]:
    """Build the multimodal layout with media and capabilities at correct positions.

    Args:
        prompt_text: The rendered prompt text with media placeholders.
        image_buffers: List of image byte buffers.
        audio_buffers: List of audio byte buffers.
        capabilities: List of capability inputs.
        content_order: Order of multimodal content parts from build_multimodal_messages.
        placeholder_token: The image placeholder token (e.g., "<|image|>").
        exclude_image_placeholder: Whether to exclude the placeholder from text segments.
        audio_placeholder: Optional audio placeholder token (e.g., "<|audio|>").
        coord_placeholder: Optional capability placeholder token (e.g., "<|coord|>").
            If provided and found in prompt_text, capabilities will be placed at
            placeholder positions instead of using content_order.

    Returns:
        List of layout segment dicts with type and length.
    """
    layout: list[dict[str, Any]] = []

    if not image_buffers and not audio_buffers and not capabilities:
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

    audio_matches = (
        list(re.finditer(re.escape(audio_placeholder), prompt_text))
        if audio_buffers and audio_placeholder
        else []
    )
    if len(audio_matches) != len(audio_buffers):
        logger.error(
            "Mismatch between rendered audio placeholders (%d) and supplied audio segments (%d).",
            len(audio_matches),
            len(audio_buffers),
        )
        raise ValueError(
            "Mismatch between audio placeholders and supplied audio parts."
        )

    capability_placeholder_token = coord_placeholder or DEFAULT_COORD_PLACEHOLDER
    capability_matches = (
        list(re.finditer(re.escape(capability_placeholder_token), prompt_text))
        if capabilities
        else []
    )
    use_placeholder_positions = bool(audio_matches or capability_matches)

    if use_placeholder_positions:
        if len(capability_matches) != len(capabilities):
            logger.error(
                "Mismatch between capability placeholders (%d) and capability parts (%d).",
                len(capability_matches),
                len(capabilities),
            )
            raise ValueError(
                "Mismatch between capability placeholders and capability parts."
            )

        all_placeholders: list[tuple[int, int, str, int]] = []

        for idx, match in enumerate(image_matches):
            all_placeholders.append((match.start(), match.end(), "image", idx))

        for idx, match in enumerate(audio_matches):
            all_placeholders.append((match.start(), match.end(), "audio", idx))

        for idx, match in enumerate(capability_matches):
            all_placeholders.append((match.start(), match.end(), "capability", idx))

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
            else:
                text_segment = prompt_text[cursor:start]

            segment_bytes = text_segment.encode("utf-8")
            if segment_bytes:
                layout.append({"type": "text", "length": len(segment_bytes)})

            # Add the placeholder content
            if ptype == "image":
                layout.append({"type": "image", "length": len(image_buffers[idx])})
            elif ptype == "audio":
                layout.append({"type": "audio", "length": len(audio_buffers[idx])})
            else:
                cap = capabilities[coord_cap_idx]
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
        audio_idx = 0
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
            elif content_type == "audio":
                layout.append(
                    {"type": "audio", "length": len(audio_buffers[audio_idx])}
                )
                audio_idx += 1
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
