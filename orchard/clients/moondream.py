from __future__ import annotations

import base64
import io
import logging
import struct
from collections.abc import Iterator
from typing import Any

from PIL import Image

from orchard.app.ipc_dispatch import IPCState
from orchard.app.model_registry import ModelRegistry
from orchard.clients.client import Client
from orchard.engine import ClientDelta

logger = logging.getLogger(__name__)

# Point (x, y) or Box (x_min, y_min, x_max, y_max)
SpatialRef = tuple[float, float] | tuple[float, float, float, float]
SpatialRefs = list[SpatialRef]


class MoondreamClient(Client):
    model_id = "moondream3"

    def __init__(self, ipc_state: IPCState, model_registry: ModelRegistry):
        super().__init__(ipc_state, model_registry)
        self.model_info = model_registry.ensure_ready_sync(self.model_id)
        self.control_tokens = self.model_info.formatter.control_tokens
        fallback_tokens = {
            "start_ground": 7,
            "placeholder": 8,
            "end_ground": 9,
            "coord": 5,
            "answer": 3,
        }
        capability_ids: dict[str, int] = fallback_tokens.copy()
        if self.model_info.capabilities:
            for name, ids in self.model_info.capabilities.items():
                if isinstance(ids, list | tuple) and ids:
                    capability_ids[name] = int(ids[0])
                elif isinstance(ids, int | float):
                    capability_ids[name] = int(ids)
        self._capability_token_ids = capability_ids

    @staticmethod
    def _image_to_data_url(image: Image.Image) -> str:
        buffer = io.BytesIO()
        format_name = (image.format or "PNG").upper()
        image.save(buffer, format=format_name)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        mime_type = f"image/{format_name.lower()}"
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def _decode_coordinate(payload_b64: str) -> float:
        raw_bytes = base64.b64decode(payload_b64)
        if len(raw_bytes) != 4:
            raise ValueError(
                f"Coordinate payload must be 4 bytes; received {len(raw_bytes)} bytes."
            )
        return float(struct.unpack("<f", raw_bytes)[0])

    def query(
        self,
        prompt: str,
        image: Image.Image | None = None,
        spatial_refs: SpatialRefs | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = []

        if image:
            data_url = self._image_to_data_url(image)
            content = [
                {"type": "input_image", "image_url": data_url},
            ]

        # Add spatial refs as capability inputs
        if spatial_refs:
            for ref in spatial_refs:
                if len(ref) == 2:
                    # Point: (x, y)
                    content.append(
                        {
                            "type": "capability",
                            "name": "coord",
                            "data": [ref[0], ref[1]],
                        }
                    )
                else:
                    # Box: (x_min, y_min, x_max, y_max) → center + size
                    x_c = (ref[0] + ref[2]) / 2
                    y_c = (ref[1] + ref[3]) / 2
                    w = ref[2] - ref[0]
                    h = ref[3] - ref[1]
                    content.append(
                        {
                            "type": "capability",
                            "name": "coord",
                            "data": [x_c, y_c],
                        }
                    )
                    content.append(
                        {
                            "type": "capability",
                            "name": "size",
                            "data": [w, h],
                        }
                    )

        content.append({"type": "input_text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        # enforce streaming to process output delta by delta
        kwargs["stream"] = True
        stream = self.chat(self.model_id, messages, **kwargs)
        if not isinstance(stream, Iterator):
            raise RuntimeError("Expected streaming iterator from chat call.")

        answer_parts: list[str] = []
        grounding: list[dict[str, Any]] = []
        reasoning_parts: list[str] = []
        current_text_parts: list[str] = []
        current_coords: list[float] = []
        in_ground_block = False
        in_answer_block = False
        ground_start_idx: int | None = None
        reasoning_text_len = 0

        def _finalize_grounding() -> None:
            nonlocal ground_start_idx
            if len(current_coords) < 2:  # Need at least one (x, y) pair
                current_text_parts.clear()
                current_coords.clear()
                ground_start_idx = None
                return

            text_block = "".join(current_text_parts)
            start_idx = ground_start_idx if ground_start_idx is not None else 0
            points = [
                (current_coords[i], current_coords[i + 1])
                for i in range(0, len(current_coords) - len(current_coords) % 2, 2)
            ]
            grounding.append(
                {
                    "start_idx": start_idx,
                    "end_idx": start_idx + len(text_block),
                    "points": points,
                }
            )
            current_text_parts.clear()
            current_coords.clear()
            ground_start_idx = None

        for delta in stream:
            assert isinstance(delta, ClientDelta)

            append_content = True
            if (
                delta.modal_decoder_id
                and delta.modal_decoder_id.endswith(".coord")
                and delta.modal_bytes_b64
                and in_ground_block
            ):
                try:
                    coord_value = self._decode_coordinate(delta.modal_bytes_b64)
                    current_coords.append(coord_value)
                    append_content = False
                except ValueError as exc:
                    logger.warning("Failed to decode coordinate payload: %s", exc)

            for token_id in delta.tokens:
                if token_id == self._capability_token_ids["start_ground"]:
                    if in_ground_block:
                        _finalize_grounding()
                    in_ground_block = True
                    ground_start_idx = reasoning_text_len
                    current_text_parts.clear()
                    append_content = False
                elif token_id == self._capability_token_ids["end_ground"]:
                    if in_ground_block:
                        _finalize_grounding()
                    in_ground_block = False
                    append_content = False
                elif token_id == self._capability_token_ids["answer"]:
                    in_answer_block = True
                    append_content = False
                elif token_id == self._capability_token_ids["placeholder"]:
                    # ignore placeholder token, i.e <|md_reserved_7|>
                    append_content = False

            if delta.content and append_content:
                delta_content = delta.content
                if in_answer_block:
                    answer_parts.append(delta_content)
                elif delta_content:
                    current_text_parts.append(delta_content)
                    reasoning_parts.append(delta_content)
                    reasoning_text_len += len(delta_content)

        if current_text_parts:
            _finalize_grounding()

        # Determine final answer: use answer_parts if available, otherwise use reasoning
        final_answer = "".join(answer_parts).strip()
        reasoning_text = "".join(reasoning_parts).strip()

        if not final_answer and reasoning_text:
            final_answer = reasoning_text

        # Build result: only include reasoning if there's grounding data
        result: dict[str, Any] = {"answer": final_answer}

        if grounding:
            result["reasoning"] = {
                "text": reasoning_text,
                "grounding": grounding,
            }

        return result

    def caption(
        self,
        image: Image.Image,
        length: str = "normal",
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate a caption for an image.

        Args:
            image: PIL Image to caption.
            length: Caption length - "normal", "short", or "long".
            stream: If True, returns a generator yielding caption chunks.
            **kwargs: Additional parameters passed to chat().

        Returns:
            {"caption": str} or {"caption": generator} if streaming.
        """
        data_url = self._image_to_data_url(image)
        messages = [
            {
                "role": "user",
                "content": [{"type": "input_image", "image_url": data_url}],
            }
        ]

        kwargs["task_name"] = f"caption_{length}"
        kwargs["stream"] = True  # Always stream internally

        response = self.chat(self.model_id, messages, **kwargs)
        if not isinstance(response, Iterator):
            raise RuntimeError("Expected streaming iterator from chat call.")

        def generator() -> Iterator[str]:
            for delta in response:
                if isinstance(delta, ClientDelta) and delta.content:
                    yield delta.content

        if stream:
            return {"caption": generator()}
        else:
            return {"caption": "".join(generator()).strip()}

    def point(
        self,
        image: Image.Image,
        object: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Find points where an object appears in an image.

        Args:
            image: PIL Image to analyze.
            object: Object to find (e.g., "dog", "face").
            **kwargs: Additional parameters passed to chat().

        Returns:
            {"points": [{"x": float, "y": float}, ...]}
        """
        data_url = self._image_to_data_url(image)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": data_url},
                    {"type": "input_text", "text": object},
                ],
            }
        ]

        kwargs["task_name"] = "point"
        kwargs["stream"] = True

        response = self.chat(self.model_id, messages, **kwargs)
        if not isinstance(response, Iterator):
            raise RuntimeError("Expected streaming iterator from chat call.")

        coords: list[float] = []
        for delta in response:
            if not isinstance(delta, ClientDelta):
                continue
            if (
                delta.modal_decoder_id
                and delta.modal_decoder_id.endswith(".coord")
                and delta.modal_bytes_b64
            ):
                try:
                    coord_value = self._decode_coordinate(delta.modal_bytes_b64)
                    coords.append(coord_value)
                except ValueError as exc:
                    logger.warning("Failed to decode coordinate: %s", exc)

        # Pair up x,y coordinates
        points = [
            {"x": coords[i], "y": coords[i + 1]} for i in range(0, len(coords) - 1, 2)
        ]
        return {"points": points}

    @staticmethod
    def _decode_size(payload_b64: str) -> tuple[float, float]:
        """Decode a size payload (width, height) from base64."""
        raw_bytes = base64.b64decode(payload_b64)
        if len(raw_bytes) != 8:
            raise ValueError(
                f"Size payload must be 8 bytes; received {len(raw_bytes)} bytes."
            )
        w, h = struct.unpack("<ff", raw_bytes)
        return float(w), float(h)

    def detect(
        self,
        image: Image.Image,
        object: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Detect objects in an image with bounding boxes.

        Args:
            image: PIL Image to analyze.
            object: Object to detect (e.g., "dog", "car").
            **kwargs: Additional parameters passed to chat().

        Returns:
            {"objects": [{"x_min", "y_min", "x_max", "y_max"}, ...]}
        """
        data_url = self._image_to_data_url(image)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": data_url},
                    {"type": "input_text", "text": object},
                ],
            }
        ]

        kwargs["task_name"] = "detect"
        kwargs["stream"] = True

        response = self.chat(self.model_id, messages, **kwargs)
        if not isinstance(response, Iterator):
            raise RuntimeError("Expected streaming iterator from chat call.")

        coords: list[float] = []
        sizes: list[tuple[float, float]] = []

        for delta in response:
            if not isinstance(delta, ClientDelta):
                continue
            if delta.modal_decoder_id and delta.modal_bytes_b64:
                if delta.modal_decoder_id.endswith(".coord"):
                    try:
                        coord_value = self._decode_coordinate(delta.modal_bytes_b64)
                        coords.append(coord_value)
                    except ValueError as exc:
                        logger.warning("Failed to decode coordinate: %s", exc)
                elif delta.modal_decoder_id.endswith(".size"):
                    try:
                        w, h = self._decode_size(delta.modal_bytes_b64)
                        sizes.append((w, h))
                    except ValueError as exc:
                        logger.warning("Failed to decode size: %s", exc)

        # Build bounding boxes from center coords + sizes
        objects = []
        num_objects = min(len(coords) // 2, len(sizes))
        for i in range(num_objects):
            x_c, y_c = coords[i * 2], coords[i * 2 + 1]
            w, h = sizes[i]
            objects.append(
                {
                    "x_min": x_c - w / 2,
                    "y_min": y_c - h / 2,
                    "x_max": x_c + w / 2,
                    "y_max": y_c + h / 2,
                }
            )
        return {"objects": objects}

    def detect_gaze(
        self,
        image: Image.Image,
        eye: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Detect where a person is looking in an image.

        Args:
            image: PIL Image to analyze.
            eye: (x, y) coordinates of the eye/face position (normalized 0-1).
            **kwargs: Additional parameters passed to chat().

        Returns:
            {"gaze": {"x": float, "y": float}} or {"gaze": None} if not detected.
        """
        if eye is None:
            raise ValueError("eye coordinates must be provided for detect_gaze")

        data_url = self._image_to_data_url(image)
        # Single coordinate capability with both x,y (8 bytes)
        # C++ MoondreamCoordCapability produces 2 embedding tokens from this input
        content: list[dict[str, Any]] = [
            {"type": "input_image", "image_url": data_url},
            {"type": "capability", "name": "coord", "data": [eye[0], eye[1]]},
        ]

        messages = [{"role": "user", "content": content}]

        kwargs["task_name"] = "detect_gaze"
        kwargs["stream"] = True

        response = self.chat(self.model_id, messages, **kwargs)
        if not isinstance(response, Iterator):
            raise RuntimeError("Expected streaming iterator from chat call.")

        coords: list[float] = []
        for delta in response:
            if not isinstance(delta, ClientDelta):
                continue
            if (
                delta.modal_decoder_id
                and delta.modal_decoder_id.endswith(".coord")
                and delta.modal_bytes_b64
            ):
                try:
                    coord_value = self._decode_coordinate(delta.modal_bytes_b64)
                    coords.append(coord_value)
                except ValueError as exc:
                    logger.warning("Failed to decode coordinate: %s", exc)

        if len(coords) >= 2:
            return {"gaze": {"x": coords[0], "y": coords[1]}}
        return {"gaze": None}
