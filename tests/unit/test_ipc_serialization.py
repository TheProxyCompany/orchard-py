import json
import struct

import pytest

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


def _payload(frame: bytes) -> bytes:
    metadata_size = struct.unpack_from("<I", frame, 0)[0]
    return frame[4 + metadata_size :]


def test_token_segments_serialize_as_type_four_with_their_ids_in_one_block() -> None:
    frame = _build_request_payload(
        request_id=1,
        model_id="test-model",
        model_path="/tmp/test-model",
        request_type="generation",
        response_channel_id=7,
        prompts=[
            {
                "prompt": "abcd",
                "layout": [
                    {"type": "text", "length": 2},
                    {"type": "tokens", "length": 3},
                    {"type": "text", "length": 2},
                    {"type": "tokens", "length": 1},
                ],
                "token_segments": [[7, 8, -9], [10]],
            }
        ],
    )

    prompt = _metadata(frame)["prompts"][0]
    payload = _payload(frame)
    assert prompt["layout_count"] == 4
    segments = [
        struct.unpack_from("<B7xQ", payload, prompt["layout_offset"] + 16 * index)
        for index in range(4)
    ]
    assert segments == [(0, 2), (4, 3), (0, 2), (4, 1)]
    assert prompt["token_data_size"] == 16
    assert struct.unpack_from("<4i", payload, prompt["token_data_offset"]) == (
        7,
        8,
        -9,
        10,
    )


def test_prompt_without_token_segments_reports_an_empty_token_block() -> None:
    frame = _build_request_payload(
        request_id=1,
        model_id="test-model",
        model_path="/tmp/test-model",
        request_type="generation",
        response_channel_id=7,
        prompts=[{"prompt": "abcd"}],
    )

    prompt = _metadata(frame)["prompts"][0]
    assert (prompt["token_data_offset"], prompt["token_data_size"]) == (0, 0)


@pytest.mark.parametrize("token_segments", [[], [[7, 8]], [[7, 8, 9], [10]]])
def test_token_segments_must_match_the_layout(token_segments: list[list[int]]) -> None:
    with pytest.raises(ValueError, match="Layout token length mismatch"):
        _build_request_payload(
            request_id=1,
            model_id="test-model",
            model_path="/tmp/test-model",
            request_type="generation",
            response_channel_id=7,
            prompts=[
                {
                    "prompt": "ab",
                    "layout": [
                        {"type": "text", "length": 2},
                        {"type": "tokens", "length": 3},
                    ],
                    "token_segments": token_segments,
                }
            ],
        )
