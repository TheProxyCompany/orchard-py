import json
import struct

from orchard.ipc.serialization import _build_request_payload


def _metadata(frame: bytes) -> dict:
    metadata_size = struct.unpack_from("<I", frame, 0)[0]
    return json.loads(frame[4 : 4 + metadata_size])


def test_empty_default_prompt_serializes_text_layout_segment() -> None:
    frame = _build_request_payload(
        request_id=1,
        model_id="test-model",
        model_path="/tmp/test-model",
        request_type="generation",
        response_channel_id=7,
        prompts=[{"prompt": ""}],
    )

    prompt = _metadata(frame)["prompts"][0]
    assert prompt["text_size"] == 0
    assert prompt["layout_count"] == 1
