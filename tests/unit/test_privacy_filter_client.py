from __future__ import annotations

import base64
import json
import struct

import pytest

from orchard.clients.privacy_filter import OpenAIPrivacyFilterClient
from orchard.engine import ClientDelta
from orchard.ipc.serialization import _build_request_payload


def test_prefill_task_request_payload_serializes_task_name() -> None:
    frame = _build_request_payload(
        request_id=7,
        model_id="openai/privacy-filter",
        model_path="/tmp/privacy-filter",
        request_type="prefill_task",
        response_channel_id=9,
        prompts=[
            {
                "prompt_bytes": b"email me at jack@example.com",
                "max_generated_tokens": 0,
                "task_name": "privacy_filter",
            }
        ],
    )

    metadata_size = struct.unpack_from("<I", frame, 0)[0]
    metadata = json.loads(frame[4 : 4 + metadata_size])

    assert metadata["request_type"] == 7
    assert metadata["prompts"][0]["task_name"] == "privacy_filter"
    assert metadata["prompts"][0]["max_generated_tokens"] == 0


def test_prefill_task_batch_request_payload_serializes_prompt_indices() -> None:
    frame = _build_request_payload(
        request_id=7,
        model_id="openai/privacy-filter",
        model_path="/tmp/privacy-filter",
        request_type="prefill_task",
        response_channel_id=9,
        prompts=[
            {
                "prompt_bytes": b"email me at jack@example.com",
                "max_generated_tokens": 0,
                "task_name": "privacy_filter",
            },
            {
                "prompt_bytes": b"hello world",
                "max_generated_tokens": 0,
                "task_name": "privacy_filter",
            },
        ],
    )

    metadata_size = struct.unpack_from("<I", frame, 0)[0]
    metadata = json.loads(frame[4 : 4 + metadata_size])

    assert metadata["request_type"] == 7
    assert [prompt["prompt_index"] for prompt in metadata["prompts"]] == [0, 1]
    assert [prompt["task_name"] for prompt in metadata["prompts"]] == [
        "privacy_filter",
        "privacy_filter",
    ]


def test_privacy_filter_client_decodes_first_modal_payload() -> None:
    payload = {
        "type": "openai_privacy_filter.token_classification",
        "token_count": 1,
        "label_ids": [0],
        "labels": ["O"],
        "scores": [0.25],
    }
    delta = ClientDelta(
        request_id=1,
        modal_decoder_id="privacy_filter",
        modal_bytes_b64=base64.b64encode(json.dumps(payload).encode()).decode(),
    )

    assert OpenAIPrivacyFilterClient._first_payload([delta]) == payload


def test_privacy_filter_client_decodes_batched_modal_payloads_in_prompt_order() -> None:
    payloads = [
        {
            "type": "openai_privacy_filter.token_classification",
            "token_count": 1,
            "label_ids": [0],
            "labels": ["O"],
            "scores": [0.25],
        },
        {
            "type": "openai_privacy_filter.token_classification",
            "token_count": 2,
            "label_ids": [13, 15],
            "labels": ["B-private_email", "E-private_email"],
            "scores": [0.5, 0.75],
        },
    ]
    deltas_by_prompt = [
        [
            ClientDelta(
                request_id=1,
                prompt_index=0,
                modal_decoder_id="privacy_filter",
                modal_bytes_b64=base64.b64encode(
                    json.dumps(payloads[0]).encode()
                ).decode(),
            )
        ],
        [
            ClientDelta(
                request_id=1,
                prompt_index=1,
                modal_decoder_id="privacy_filter",
                modal_bytes_b64=base64.b64encode(
                    json.dumps(payloads[1]).encode()
                ).decode(),
            )
        ],
    ]

    assert OpenAIPrivacyFilterClient._payloads_by_prompt(deltas_by_prompt) == payloads


def test_privacy_filter_client_caps_v1_input_size() -> None:
    with pytest.raises(ValueError, match="callers must chunk larger inputs"):
        OpenAIPrivacyFilterClient._check_input_size(
            "x" * (OpenAIPrivacyFilterClient.max_input_bytes + 1)
        )
