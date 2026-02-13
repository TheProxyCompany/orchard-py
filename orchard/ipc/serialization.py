from __future__ import annotations

import json
import struct
from collections.abc import Mapping, Sequence
from typing import Any

_PAYLOAD_ALIGNMENT = 16
_LAYOUT_SEGMENT_STRUCT = struct.Struct("<B7xQ")
_IMAGE_SPAN_STRUCT = struct.Struct("<Q")

# Segment type codes matching C++ SerializedSegmentType
_SEGMENT_TYPE_TEXT = 0
_SEGMENT_TYPE_IMAGE = 1
_SEGMENT_TYPE_CAPABILITY = 2

_REQUEST_TYPE_CODES = {
    "generation": 0,
    "embedding": 1,
    "query": 2,
    "point": 3,
    "detect": 4,
    "agent": 5,
    "omni": 6,
}

__all__ = ["_build_request_payload"]

_METADATA_PREFIX_STRUCT = struct.Struct("<I")


def _align_offset(value: int, alignment: int = _PAYLOAD_ALIGNMENT) -> int:
    remainder = value % alignment
    if remainder == 0:
        return value
    return value + (alignment - remainder)


def _coerce_bytes(value: Any) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    return str(value).encode("utf-8")


def _encode_image_buffers(buffers: Sequence[bytes]) -> tuple[bytes, int, bytes]:
    if not buffers:
        return b"", 0, b""
    span_buffer = bytearray(len(buffers) * _IMAGE_SPAN_STRUCT.size)
    payload_buffer = bytearray(sum(len(entry) for entry in buffers))
    data_cursor = 0
    for idx, entry in enumerate(buffers):
        _IMAGE_SPAN_STRUCT.pack_into(
            span_buffer, idx * _IMAGE_SPAN_STRUCT.size, len(entry)
        )
        payload_buffer[data_cursor : data_cursor + len(entry)] = entry
        data_cursor += len(entry)
    return bytes(span_buffer), len(buffers), bytes(payload_buffer)


def _encode_capabilities(
    capabilities: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], bytes]:
    """Encode capability entries into JSON metadata and binary payload.

    Returns:
        Tuple of (capability_metadata_list, capability_data_bytes)
    """
    if not capabilities:
        return [], b""

    metadata_list: list[Mapping[str, Any]] = []
    data_buffer = bytearray()

    for cap in capabilities:
        name = str(cap.get("name", ""))
        payload = _coerce_bytes(cap.get("payload", b""))
        position = int(cap.get("position", 0))

        metadata_list.append(
            {
                "name": name,
                "position": position,
                "payload_size": len(payload),
            }
        )
        data_buffer.extend(payload)

    return metadata_list, bytes(data_buffer)


def _encode_layout(
    layout: Sequence[Mapping[str, Any]], text_len: int, image_buffers: Sequence[bytes]
) -> tuple[bytes, int]:
    """Encode layout segments including text, image, and capability types."""
    segments: list[tuple[int, int]] = []
    if not layout:
        if text_len:
            segments.append((_SEGMENT_TYPE_TEXT, text_len))
        for image in image_buffers:
            segments.append((_SEGMENT_TYPE_IMAGE, len(image)))
    else:
        for segment in layout:
            seg_type = str(segment.get("type", "text")).lower()
            length = int(segment.get("length", 0))
            if seg_type == "text":
                segments.append((_SEGMENT_TYPE_TEXT, length))
            elif seg_type == "image":
                segments.append((_SEGMENT_TYPE_IMAGE, length))
            elif seg_type == "capability":
                # Capability segments use length=0 in binary layout (actual data is in JSON)
                segments.append((_SEGMENT_TYPE_CAPABILITY, 0))
            else:
                raise ValueError(f"Unsupported layout segment type: {seg_type}")

    if not segments:
        return b"", 0

    layout_text_bytes = sum(
        length for seg_type, length in segments if seg_type == _SEGMENT_TYPE_TEXT
    )
    layout_image_bytes = sum(
        length for seg_type, length in segments if seg_type == _SEGMENT_TYPE_IMAGE
    )
    total_image_bytes = sum(len(image) for image in image_buffers)

    if layout_text_bytes != text_len:
        raise ValueError(
            f"Layout text length mismatch (expected {text_len}, got {layout_text_bytes})."
        )
    if layout_image_bytes != total_image_bytes:
        raise ValueError(
            "Layout image length mismatch "
            f"(expected {total_image_bytes}, got {layout_image_bytes})."
        )

    buffer = bytearray(len(segments) * _LAYOUT_SEGMENT_STRUCT.size)
    for idx, (seg_type, length) in enumerate(segments):
        _LAYOUT_SEGMENT_STRUCT.pack_into(
            buffer,
            idx * _LAYOUT_SEGMENT_STRUCT.size,
            seg_type,
            length,
        )
    return bytes(buffer), len(segments)


def _normalise_request_type(request_type: str | int) -> int:
    if isinstance(request_type, int):
        return request_type
    code = _REQUEST_TYPE_CODES.get(str(request_type).lower())
    if code is None:
        raise ValueError(f"Unsupported request type: {request_type}")
    return code


def _build_request_payload(
    *,
    request_id: int,
    model_id: str,
    model_path: str,
    request_type: str | int,
    response_channel_id: int,
    prompts: Sequence[Mapping[str, Any]],
    request_channel_id: int = 0,
    parent_request_id: int | None = None,
) -> bytes:
    if not prompts:
        raise ValueError("At least one prompt payload is required.")

    prompt_payloads = list(prompts)
    parent_id = parent_request_id or request_id
    metadata = {
        "request_id": int(parent_id),
        "model_id": model_id,
        "model_path": str(model_path),
        "request_type": _normalise_request_type(request_type),
        "request_channel_id": int(request_channel_id),
        "response_channel_id": int(response_channel_id),
    }
    metadata_prompts: list[dict[str, Any]] = []
    metadata["prompts"] = metadata_prompts

    total_size = 0

    blob_fragments: list[tuple[int, bytes]] = []

    for index, prompt in enumerate(prompt_payloads):
        text_bytes = prompt.get("prompt_bytes")
        if text_bytes is None:
            text_bytes = prompt.get("prompt")
        text_buffer = _coerce_bytes(text_bytes)

        image_buffers_raw = [
            _coerce_bytes(buffer) for buffer in prompt.get("image_buffers", [])
        ]
        image_span_bytes, image_count, image_data_bytes = _encode_image_buffers(
            image_buffers_raw
        )

        # Encode capability entries
        capability_metadata, capability_data_bytes = _encode_capabilities(
            prompt.get("capabilities", [])
        )

        layout_bytes, layout_count = _encode_layout(
            prompt.get("layout", []), len(text_buffer), image_buffers_raw
        )

        def reserve_blob(data: bytes) -> tuple[int, int]:
            nonlocal total_size
            if not data:
                return 0, 0
            total_size = _align_offset(total_size)
            offset = total_size
            blob_fragments.append((offset, data))
            total_size += len(data)
            return offset, len(data)

        text_offset, text_size = reserve_blob(text_buffer)
        image_sizes_offset, _ = reserve_blob(image_span_bytes)
        image_data_offset, image_data_size = reserve_blob(image_data_bytes)
        capability_data_offset, capability_data_size = reserve_blob(
            capability_data_bytes
        )
        layout_offset, _ = reserve_blob(layout_bytes)

        stop_sequences_raw = prompt.get("stop_sequences") or []
        stop_sequences: list[str] = []
        for sequence in stop_sequences_raw:
            if isinstance(sequence, str):
                stop_sequences.append(sequence)
            else:
                stop_sequences.append(_coerce_bytes(sequence).decode("utf-8"))

        tool_schemas_value = prompt.get("tool_schemas_json", "")
        if isinstance(tool_schemas_value, str):
            tool_schemas_str = tool_schemas_value
        else:
            tool_schemas_str = _coerce_bytes(tool_schemas_value).decode("utf-8")

        tool_calling_tokens_raw = prompt.get("tool_calling_tokens") or {}
        raw_formats = tool_calling_tokens_raw.get("formats") or []
        formats: list[dict[str, str]] = []
        for format_entry in raw_formats:
            if not isinstance(format_entry, dict):
                continue
            formats.append(
                {
                    "call_start": str(format_entry.get("call_start", "")),
                    "call_end": str(format_entry.get("call_end", "")),
                }
            )

        tool_calling_tokens = {
            "formats": formats,
            "section_start": str(tool_calling_tokens_raw.get("section_start", "")),
            "section_end": str(tool_calling_tokens_raw.get("section_end", "")),
        }
        max_tool_calls = int(prompt.get("max_tool_calls") or 0)

        tool_choice_value = prompt.get("tool_choice", "auto")
        if isinstance(tool_choice_value, dict):
            tool_choice_str = json.dumps(tool_choice_value)
        elif isinstance(tool_choice_value, str):
            tool_choice_str = tool_choice_value
        else:
            tool_choice_str = str(tool_choice_value)

        response_format_value = prompt.get("response_format_json", "")
        if isinstance(response_format_value, str):
            response_format_str = response_format_value
        else:
            response_format_str = _coerce_bytes(response_format_value).decode("utf-8")

        task_name_value = prompt.get("task_name")
        if isinstance(task_name_value, str) or task_name_value is None:
            task_name_str = task_name_value
        else:
            task_name_str = _coerce_bytes(task_name_value).decode("utf-8")

        reasoning_effort_value = prompt.get("reasoning_effort")
        if isinstance(reasoning_effort_value, str) or reasoning_effort_value is None:
            reasoning_effort_str = reasoning_effort_value
        else:
            reasoning_effort_str = _coerce_bytes(reasoning_effort_value).decode("utf-8")

        logits_params = prompt.get("logits_params", {})

        sampling_params = prompt.get("sampling_params", {})
        temperature = float(sampling_params.get("temperature", 1.0))
        top_p = float(sampling_params.get("top_p", 1.0))
        top_k = int(sampling_params.get("top_k", -1))
        min_p = float(sampling_params.get("min_p", 0.0))
        rng_seed = int(sampling_params.get("rng_seed", 0)) & 0xFFFFFFFF

        top_logprobs = int(logits_params.get("top_logprobs", 0))
        frequency_penalty = float(logits_params.get("frequency_penalty", 0.0))
        presence_penalty = float(logits_params.get("presence_penalty", 0.0))
        repetition_context_size = int(logits_params.get("repetition_context_size", 0))
        repetition_penalty = float(logits_params.get("repetition_penalty", 1.0))

        max_generated_tokens = int(prompt.get("max_generated_tokens", 0))
        num_candidates = max(1, int(prompt.get("num_candidates", 1)))
        best_of = int(prompt.get("best_of", num_candidates))
        final_candidates = int(prompt.get("final_candidates", best_of))

        logit_bias_entries = [
            {"token": int(token), "bias": float(bias)}
            for token, bias in logits_params.get("logit_bias", {}).items()
        ]

        metadata_prompts.append(
            {
                "prompt_index": index,
                "num_candidates": num_candidates,
                "best_of": best_of,
                "final_candidates": final_candidates,
                "max_generated_tokens": max_generated_tokens,
                "text_offset": text_offset,
                "text_size": text_size,
                "image_data_offset": image_data_offset,
                "image_data_size": image_data_size,
                "image_sizes_offset": image_sizes_offset,
                "image_count": image_count,
                "capability_data_offset": capability_data_offset,
                "capability_data_size": capability_data_size,
                "capabilities": capability_metadata,
                "layout_offset": layout_offset,
                "layout_count": layout_count,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "rng_seed": rng_seed,
                "top_logprobs": top_logprobs,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "repetition_context_size": repetition_context_size,
                "repetition_penalty": repetition_penalty,
                "stop_sequences": stop_sequences,
                "tool_schemas_json": tool_schemas_str,
                "tool_calling_tokens": tool_calling_tokens,
                "tool_choice": tool_choice_str,
                "max_tool_calls": max_tool_calls,
                "response_format_json": response_format_str,
                "logit_bias": logit_bias_entries,
                "task_name": task_name_str,
                "reasoning_effort": reasoning_effort_str,
            }
        )

    payload = bytearray(total_size)

    for offset, data in blob_fragments:
        payload[offset : offset + len(data)] = data

    payload_bytes = bytes(payload)
    metadata_bytes = json.dumps(
        metadata, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")

    metadata_len = len(metadata_bytes)
    if metadata_len > 0xFFFFFFFF:
        raise ValueError("Metadata section exceeds 4-byte length prefix capacity.")

    total_length = _METADATA_PREFIX_STRUCT.size + metadata_len + len(payload_bytes)
    framed = bytearray(total_length)
    _METADATA_PREFIX_STRUCT.pack_into(framed, 0, metadata_len)
    framed[
        _METADATA_PREFIX_STRUCT.size : _METADATA_PREFIX_STRUCT.size + metadata_len
    ] = metadata_bytes
    framed[_METADATA_PREFIX_STRUCT.size + metadata_len :] = payload_bytes
    return bytes(framed)
