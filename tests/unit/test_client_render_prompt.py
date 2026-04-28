from __future__ import annotations

from typing import Any

import pytest

import orchard.clients.client as client_module
from orchard.app.model_registry import ModelInfo
from orchard.clients.client import Client
from orchard.clients.responses import ResponsesRequest

DATA_URL = "data:image/png;base64,AA=="


class _FakeRoles:
    def model_dump(self) -> dict[str, str]:
        return {
            "system": "<system>",
            "user": "<user>",
            "agent": "<agent>",
            "tool": "<tool>",
        }


class _FakeControlTokens:
    roles = _FakeRoles()


class _FakeFormatter:
    control_tokens = _FakeControlTokens()
    image_placeholder = "<|image|>"
    should_clip_image_placeholder = True

    def apply_template(
        self,
        messages: list[dict[str, Any]],
        *,
        reasoning: bool = False,
        task: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        rendered_messages: list[str] = []
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                rendered_content = "".join(self._render_part(part) for part in content)
            else:
                rendered_content = str(content)
            rendered_messages.append(f"{message['role']}:{rendered_content}")

        suffix: list[str] = []
        if reasoning:
            suffix.append("reasoning=1")
        if task:
            suffix.append(f"task={task}")
        if tools:
            suffix.append(f"tools={len(tools)}")
        if suffix:
            rendered_messages.append(",".join(suffix))
        return " | ".join(rendered_messages)

    def get_coord_placeholder(self) -> str:
        return "<|coord|>"

    def get_tool_calling_tokens(self) -> dict[str, Any]:
        return {
            "formats": [],
            "section_start": "<tool>",
            "section_end": "</tool>",
        }

    def get_thinking_tokens(self) -> dict[str, str]:
        return {"start": "<think>\n", "end": "\n</think>"}

    def _render_part(self, part: Any) -> str:
        part_type = part["type"]
        if part_type == "image":
            return self.image_placeholder
        if part_type == "capability":
            return self.get_coord_placeholder()
        return str(part)


class _FakeRegistry:
    def __init__(self, info: ModelInfo) -> None:
        self._info = info

    async def get_info(self, model_id: str) -> ModelInfo:
        assert model_id == self._info.model_id
        return self._info

    def ensure_ready_sync(self, model_id: str) -> ModelInfo:
        assert model_id == self._info.model_id
        return self._info


class _DummySocket:
    async def asend(self, payload: bytes) -> None:
        self.last_payload = payload


class _DummyIPCState:
    def __init__(self) -> None:
        self.engine_dead = False
        self.response_channel_id = None
        self.active_request_queues: dict[int, Any] = {}
        self.request_socket = _DummySocket()
        self._next_request_id = 0

    async def get_next_request_id(self) -> int:
        self._next_request_id += 1
        return self._next_request_id


def _make_client() -> Client:
    info = ModelInfo(
        model_id="test-model",
        model_path="/models/test-model",
        formatter=_FakeFormatter(),
    )
    return Client(_DummyIPCState(), _FakeRegistry(info))


@pytest.mark.asyncio
async def test_arender_prompt_matches_submit_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": DATA_URL},
                {"type": "input_text", "text": "Describe the image."},
            ],
        }
    ]
    kwargs = {
        "instructions": "Be brief.",
        "temperature": 0.25,
        "top_p": 0.8,
        "top_k": 7,
        "min_p": 0.1,
        "rng_seed": 1234,
        "max_generated_tokens": 64,
        "top_logprobs": 3,
        "frequency_penalty": 0.2,
        "presence_penalty": 0.3,
        "repetition_context_size": 42,
        "repetition_penalty": 1.15,
        "logit_bias": {"7": -1.25},
        "stop": ["END"],
        "n": 2,
        "best_of": 4,
        "final_candidates": 2,
        "task_name": "point",
        "response_format": {"type": "json_object"},
        "tool_choice": {"type": "function", "name": "lookup"},
        "core_tools": [{"name": "lookup", "type": "object", "properties": {}}],
        "max_tool_calls": 3,
        "reasoning_effort": "medium",
    }

    captured: dict[str, Any] = {}

    def _capture_request_payload(**payload_kwargs: Any) -> bytes:
        captured["prompt_payload"] = payload_kwargs["prompts"][0]
        return b"captured"

    monkeypatch.setattr(
        client_module, "_build_request_payload", _capture_request_payload
    )

    rendered = await client.arender_prompt("test-model", messages, **kwargs)
    await client._asubmit_request(1, "test-model", messages, **kwargs)  # noqa: SLF001

    private_payload = captured["prompt_payload"]
    assert rendered["rendered_prompt_text"] == private_payload["prompt_bytes"].decode(
        "utf-8"
    )
    assert rendered["model_path"] == "/models/test-model"
    assert "<|image|>" not in rendered["rendered_prompt_text"]
    assert rendered["sampling_params"] == {
        "temperature": 0.25,
        "top_p": 0.8,
        "top_k": 7,
        "min_p": 0.1,
        "rng_seed": 1234,
        "top_logprobs": 3,
        "frequency_penalty": 0.2,
        "presence_penalty": 0.3,
        "repetition_context_size": 42,
        "repetition_penalty": 1.15,
        "logit_bias": {7: -1.25},
        "n": 2,
        "best_of": 4,
        "final_candidates": 2,
    }
    assert private_payload["sampling_params"] == {
        "temperature": 0.25,
        "top_p": 0.8,
        "top_k": 7,
        "min_p": 0.1,
        "rng_seed": 1234,
    }
    assert private_payload["logits_params"] == {
        "top_logprobs": 3,
        "frequency_penalty": 0.2,
        "presence_penalty": 0.3,
        "repetition_context_size": 42,
        "repetition_penalty": 1.15,
        "logit_bias": {7: -1.25},
    }
    assert rendered["tool_choice"] == {"type": "function", "name": "lookup"}
    assert rendered["tool_schemas_json"] == (
        '[{"name": "lookup", "type": "object", "properties": {}}]'
    )
    assert private_payload["active_tool_schemas_json"] == rendered["tool_schemas_json"]
    assert private_payload["thinking_tokens"] == {
        "start": "<think>\n",
        "end": "\n</think>",
    }


@pytest.mark.asyncio
async def test_arender_prompt_splits_core_and_active_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client()
    messages = [{"role": "user", "content": "Call the tool."}]
    kwargs = {
        "core_tools": [{"name": "search_tools", "type": "object", "properties": {}}],
        "active_tools": [
            {"name": "search_tools", "type": "object", "properties": {}},
            {"name": "loaded_tool", "type": "object", "properties": {}},
        ],
    }

    captured: dict[str, Any] = {}

    def _capture_request_payload(**payload_kwargs: Any) -> bytes:
        captured["prompt_payload"] = payload_kwargs["prompts"][0]
        return b"captured"

    monkeypatch.setattr(
        client_module, "_build_request_payload", _capture_request_payload
    )

    rendered = await client.arender_prompt("test-model", messages, **kwargs)
    await client._asubmit_request(1, "test-model", messages, **kwargs)  # noqa: SLF001

    private_payload = captured["prompt_payload"]
    assert "tools=1" in rendered["rendered_prompt_text"]
    assert private_payload["tool_schemas_json"] == (
        '[{"name": "search_tools", "type": "object", "properties": {}}]'
    )
    assert private_payload["active_tool_schemas_json"] == (
        '[{"name": "loaded_tool", "type": "object", "properties": {}}, '
        '{"name": "search_tools", "type": "object", "properties": {}}]'
    )


@pytest.mark.asyncio
async def test_arender_responses_prompt_matches_submit_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client()
    request = ResponsesRequest.from_text(
        "Call the tool.",
        instructions="You are concise.",
        temperature=0.0,
        top_p=0.9,
        top_k=5,
        min_p=0.2,
        max_output_tokens=32,
        top_logprobs=4,
        frequency_penalty=0.1,
        presence_penalty=0.4,
        tool_choice="required",
        core_tools=[
            {
                "type": "function",
                "name": "lookup",
                "description": "Lookup tool",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
    )

    captured: dict[str, Any] = {}

    def _capture_request_payload(**payload_kwargs: Any) -> bytes:
        captured["prompt_payload"] = payload_kwargs["prompts"][0]
        return b"captured"

    monkeypatch.setattr(
        client_module, "_build_request_payload", _capture_request_payload
    )

    rendered = await client.arender_responses_prompt("test-model", request=request)
    await client._asubmit_request(  # noqa: SLF001
        1,
        "test-model",
        request.to_messages(),
        **request.to_submit_kwargs(),
    )

    private_payload = captured["prompt_payload"]
    assert rendered["rendered_prompt_text"] == private_payload["prompt_bytes"].decode(
        "utf-8"
    )
    assert rendered["sampling_params"]["temperature"] == 0.0
    assert rendered["sampling_params"]["top_p"] == 0.9
    assert rendered["sampling_params"]["top_k"] == 5
    assert rendered["sampling_params"]["min_p"] == 0.2
    assert rendered["sampling_params"]["top_logprobs"] == 4
    assert rendered["sampling_params"]["frequency_penalty"] == 0.1
    assert rendered["sampling_params"]["presence_penalty"] == 0.4
    assert rendered["max_generated_tokens"] == 32
    assert rendered["tool_choice"] == "required"
    assert private_payload["tool_schemas_json"] == (
        '[{"name": "lookup", "type": "object", "description": "Lookup tool", '
        '"properties": {"name": {"const": "lookup"}, "arguments": {"type": "object", '
        '"properties": {}}}, "strict": true, "required": ["name", "arguments"]}]'
    )
    assert (
        private_payload["active_tool_schemas_json"]
        == private_payload["tool_schemas_json"]
    )
    assert rendered["task_name"] is None
    assert rendered["reasoning_effort"] is None
    assert private_payload["thinking_tokens"] == {
        "start": "<think>\n",
        "end": "\n</think>",
    }
