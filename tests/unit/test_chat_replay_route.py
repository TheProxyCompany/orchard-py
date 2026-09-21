"""`reasoning_content` and `generation` over the OpenAI-compatible chat route."""

from __future__ import annotations

import json
import struct
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from orchard.app.model_registry import ModelInfo, ModelLoadState
from orchard.server.routes.chat import chat_router

CANONICAL_ID = "org/test-model"


class _Formatter:
    """Renders what the chat templates render for each kind of assistant message."""

    def supports_native_thinking(self) -> bool:
        return True

    def apply_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        rendered = []
        for message in messages:
            if "generated" in message:
                body = (
                    f"<thinking={message['generated_thinking']}>{message['generated']}"
                )
            elif message.get("reasoning_content"):
                body = (
                    f"<think>{message['reasoning_content']}</think>{message['content']}"
                )
            else:
                body = message["content"]
            rendered.append(f"{message['role']}:{body}")
        return " | ".join(rendered)

    def get_generation_defaults(self, profile: str = "default") -> dict[str, Any]:
        return {}

    def get_tool_calling_tokens(self) -> dict[str, Any]:
        return {"formats": [], "section_start": "", "section_end": ""}

    def get_output_frame_tokens(self) -> dict[str, str]:
        return {}

    def get_thinking_tokens(self) -> dict[str, str]:
        return {"start": "<think>", "end": "</think>"}


class _Registry:
    def __init__(self, info: ModelInfo) -> None:
        self._info = info

    async def schedule_model(self, requested: str) -> tuple[ModelLoadState, str]:
        return ModelLoadState.READY, CANONICAL_ID

    def get_if_ready(self, model_id: str) -> ModelInfo:
        return self._info


class _IPCState:
    """Answers every request with one reasoning delta and one final message delta."""

    response_channel_id = None

    def __init__(self) -> None:
        self.active_request_queues: dict[int, Any] = {}
        self.frames: list[bytes] = []

    async def get_next_request_id(self) -> int:
        return 1

    async def send_request(self, frame: bytes) -> None:
        self.frames.append(frame)
        queue = self.active_request_queues[1].queue
        base = {"request_id": 1, "prompt_index": 0, "candidate_index": 0}
        reasoning = {
            "item_type": "reasoning",
            "event_type": "content_delta",
            "identifier": "reasoning",
            "delta": "why",
        }
        answer = {"item_type": "message", "event_type": "content_delta", "delta": "ok"}
        await queue.put(
            {**base, "tokens": [11, 12], "content": "why", "state_events": [reasoning]}
        )
        await queue.put(
            {
                **base,
                "tokens": [13],
                "content": "ok",
                "state_events": [answer],
                "is_final_delta": True,
                "finish_reason": "stop",
            }
        )

    async def next_delta(self, queue: Any) -> dict[str, Any]:
        return await queue.get()

    async def cancel_request(self, request_id: int) -> dict[str, Any]:
        return {"status": "accepted"}


def _serve(capabilities: dict[str, list[int]] | None) -> tuple[TestClient, _IPCState]:
    app = FastAPI()
    app.include_router(chat_router, prefix="/v1")
    app.state.ipc_state = _IPCState()
    app.state.model_registry = _Registry(
        ModelInfo(
            model_id=CANONICAL_ID,
            model_path="/models/test-model",
            formatter=_Formatter(),
            capabilities=capabilities,
        )
    )
    return TestClient(app), app.state.ipc_state


def _ask(client: TestClient, messages: list[dict[str, Any]], **fields: Any) -> Any:
    return client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "temperature": 1.0,
            "messages": messages,
            **fields,
        },
    )


def _prompt(frame: bytes) -> tuple[dict[str, Any], bytes]:
    metadata_size = struct.unpack_from("<I", frame, 0)[0]
    metadata = json.loads(frame[4 : 4 + metadata_size])
    return metadata["prompts"][0], frame[4 + metadata_size :]


def _text(frame: bytes) -> str:
    prompt, payload = _prompt(frame)
    start = prompt["text_offset"]
    return payload[start : start + prompt["text_size"]].decode("utf-8")


def test_the_returned_message_can_be_sent_back_and_is_replayed_as_token_ids() -> None:
    client, ipc_state = _serve({"token_segments": [1]})
    asked = [{"role": "user", "content": "one"}]

    first = _ask(client, asked)
    assert first.status_code == 200
    reply = first.json()["choices"][0]["message"]
    assert reply == {
        "role": "assistant",
        "content": "ok",
        "reasoning_content": "why",
        "generation": {
            "model": CANONICAL_ID,
            "tokens": [11, 12, 13],
            "thinking": True,
        },
    }

    second = _ask(client, [*asked, reply, {"role": "user", "content": "two"}])
    assert second.status_code == 200
    prompt, payload = _prompt(ipc_state.frames[1])
    assert (
        _text(ipc_state.frames[1]) == "user:one | assistant:<thinking=True> | user:two"
    )
    assert prompt["layout_count"] == 3
    assert struct.unpack_from("<3i", payload, prompt["token_data_offset"]) == (
        11,
        12,
        13,
    )


def test_a_reply_renders_as_text_with_its_reasoning_when_it_cannot_be_replayed() -> (
    None
):
    client, ipc_state = _serve(None)
    reply = {
        "role": "assistant",
        "content": "ok",
        "reasoning_content": "why",
        "generation": {"model": CANONICAL_ID, "tokens": [11, 12, 13], "thinking": True},
    }

    response = _ask(client, [{"role": "user", "content": "one"}, reply])

    assert response.status_code == 200
    prompt, _ = _prompt(ipc_state.frames[0])
    assert _text(ipc_state.frames[0]) == "user:one | assistant:<think>why</think>ok"
    assert prompt["layout_count"] == 1
    assert prompt["token_data_size"] == 0


def test_a_stream_ends_each_reply_with_its_generation_record() -> None:
    client, _ = _serve({"token_segments": [1]})

    response = _ask(
        client, [{"role": "user", "content": "one"}], stream=True, reasoning=False
    )

    assert response.status_code == 200
    chunks = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: {")
    ]
    deltas = [chunk["choices"][0]["delta"] for chunk in chunks]
    assert deltas[:-1] == [
        {"role": "assistant", "content": "why"},
        {"content": "ok"},
    ]
    assert deltas[-1] == {
        "generation": {
            "model": CANONICAL_ID,
            "tokens": [11, 12, 13],
            "thinking": False,
        }
    }
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
