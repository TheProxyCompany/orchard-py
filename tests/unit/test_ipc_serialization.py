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
        lossless_responses=False,
        prompts=[{"prompt": ""}],
    )

    prompt = _metadata(frame)["prompts"][0]
    assert prompt["text_size"] == 0
    assert prompt["layout_count"] == 1


def _embedding_request(*, lossless_responses: bool) -> dict:
    return _metadata(
        _build_request_payload(
            request_id=1,
            model_id="test-model",
            model_path="/tmp/test-model",
            request_type="embedding",
            response_channel_id=7,
            lossless_responses=lossless_responses,
            prompts=[{"prompt": "hello"}],
        )
    )


def test_a_request_asks_for_the_lossless_response_route_only_when_told_to() -> None:
    # The engine compares this exact string; anything else means
    # publish/subscribe.
    assert _embedding_request(lossless_responses=True)["response_transport"] == (
        "pull_v1"
    )
    # Without it the request is the one every engine knows, field for field.
    assert "response_transport" not in _embedding_request(lossless_responses=False)
