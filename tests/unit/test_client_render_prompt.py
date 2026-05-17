from __future__ import annotations

import json
from typing import Any

import pytest

import orchard.clients.client as client_module
from orchard.app.model_registry import ModelInfo
from orchard.clients.client import Client
from orchard.clients.responses import ResponsesRequest
from orchard.engine import ClientDelta
from orchard.formatter.formatter import ChatFormatter

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

    def supports_native_thinking(self) -> bool:
        return True

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


def _make_client(formatter: Any | None = None) -> Client:
    info = ModelInfo(
        model_id="test-model",
        model_path="/models/test-model",
        formatter=formatter or _FakeFormatter(),
    )
    return Client(_DummyIPCState(), _FakeRegistry(info))


def test_chat_aggregate_uses_message_state_events_when_present() -> None:
    client = _make_client()
    deltas = [
        ClientDelta(
            request_id=1,
            content="hidden thought",
            state_events=[
                {
                    "event_type": "content_delta",
                    "item_type": "reasoning",
                    "identifier": "reasoning",
                    "delta": "hidden thought",
                }
            ],
        ),
        ClientDelta(
            request_id=1,
            content="visible answer",
            state_events=[
                {
                    "event_type": "content_delta",
                    "item_type": "message",
                    "identifier": "message",
                    "delta": "visible answer",
                }
            ],
        ),
        ClientDelta(request_id=1, is_final_delta=True, finish_reason="stop"),
    ]

    response = client._aggregate_response(deltas)  # noqa: SLF001
    assert response.text == "visible answer"
    assert response.reasoning == ["hidden thought"]


def test_chat_aggregate_does_not_duplicate_completed_message_value() -> None:
    client = _make_client()
    deltas = [
        ClientDelta(
            request_id=1,
            content='{"answer":"A"}',
            state_events=[
                {
                    "event_type": "content_delta",
                    "item_type": "message",
                    "identifier": "message",
                    "delta": '{"answer":"A"}',
                }
            ],
        ),
        ClientDelta(
            request_id=1,
            state_events=[
                {
                    "event_type": "item_completed",
                    "item_type": "message",
                    "identifier": "message",
                    "value": '{"answer":"A"}',
                }
            ],
        ),
        ClientDelta(request_id=1, is_final_delta=True, finish_reason="stop"),
    ]

    assert client._aggregate_response(deltas).text == '{"answer":"A"}'  # noqa: SLF001


def test_chat_aggregate_uses_completed_message_when_no_delta_seen() -> None:
    client = _make_client()
    deltas = [
        ClientDelta(
            request_id=1,
            state_events=[
                {
                    "event_type": "item_completed",
                    "item_type": "message",
                    "identifier": "message",
                    "value": "complete",
                }
            ],
        ),
    ]

    assert client._aggregate_response(deltas).text == "complete"  # noqa: SLF001


def test_chat_aggregate_keeps_raw_content_without_state_events() -> None:
    client = _make_client()
    deltas = [
        ClientDelta(request_id=1, content="hello "),
        ClientDelta(request_id=1, content="world"),
    ]

    assert client._aggregate_response(deltas).text == "hello world"  # noqa: SLF001


def test_chat_aggregate_exposes_tool_calls_from_state_events() -> None:
    client = _make_client()
    deltas = [
        ClientDelta(
            request_id=1,
            state_events=[
                {
                    "event_type": "content_delta",
                    "item_type": "tool_call",
                    "output_index": 0,
                    "identifier": "arguments",
                    "delta": '{"content":"hi"}',
                },
                {
                    "event_type": "item_completed",
                    "item_type": "tool_call",
                    "output_index": 0,
                    "identifier": "tool_call:share_to_party",
                    "value": {
                        "name": "share_to_party",
                        "arguments": {"content": "hi"},
                    },
                },
            ],
        )
    ]

    response = client._aggregate_response(deltas)  # noqa: SLF001

    assert response.text == ""
    assert response.tool_calls == [
        {"name": "share_to_party", "arguments": {"content": "hi"}}
    ]


def test_chat_aggregate_parses_tool_call_json_string_completion() -> None:
    client = _make_client()
    deltas = [
        ClientDelta(
            request_id=1,
            state_events=[
                {
                    "event_type": "content_delta",
                    "item_type": "tool_call",
                    "output_index": 0,
                    "identifier": "arguments",
                    "delta": '{content:<|"|>hi<|"|>}',
                },
                {
                    "event_type": "item_completed",
                    "item_type": "tool_call",
                    "output_index": 0,
                    "identifier": "tool_call:share_to_party",
                    "value": '{"name":"share_to_party","arguments":{"content":"hi"}}',
                },
            ],
        )
    ]

    response = client._aggregate_response(deltas)  # noqa: SLF001

    assert response.tool_calls == [
        {"name": "share_to_party", "arguments": {"content": "hi"}}
    ]


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
        "min_tool_calls": 2,
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
        "deterministic": False,
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
        "deterministic": False,
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
async def test_batched_prompt_payload_uses_per_prompt_response_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client()
    conversations = [
        [{"role": "user", "content": "Return city JSON."}],
        [{"role": "user", "content": "Return animal JSON."}],
    ]
    response_formats = [
        {
            "type": "json_schema",
            "json_schema": {"name": "city", "schema": {"type": "object"}},
        },
        {
            "type": "json_schema",
            "json_schema": {"name": "animal", "schema": {"type": "array"}},
        },
    ]
    captured: dict[str, Any] = {}

    def _capture_request_payload(**payload_kwargs: Any) -> bytes:
        captured["prompts"] = payload_kwargs["prompts"]
        return b"captured"

    monkeypatch.setattr(
        client_module, "_build_request_payload", _capture_request_payload
    )

    await client._asubmit_request_batch(  # noqa: SLF001
        1,
        "test-model",
        conversations,
        response_format=response_formats,
    )

    assert len(captured["prompts"]) == 2
    assert (
        json.loads(captured["prompts"][0]["response_format_json"])
        == response_formats[0]
    )
    assert (
        json.loads(captured["prompts"][1]["response_format_json"])
        == response_formats[1]
    )


@pytest.mark.asyncio
async def test_arender_responses_prompt_maps_tools_to_core_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client()
    request = ResponsesRequest.from_text(
        "Call the tool.",
        tools=[
            {
                "type": "function",
                "name": "lookup",
                "description": "Lookup tool",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
        tool_choice="required",
        min_tool_calls=1,
        max_tool_calls=1,
    )

    captured: dict[str, Any] = {}

    def _capture_request_payload(**payload_kwargs: Any) -> bytes:
        captured["prompt_payload"] = payload_kwargs["prompts"][0]
        return b"captured"

    monkeypatch.setattr(
        client_module, "_build_request_payload", _capture_request_payload
    )

    await client._asubmit_request(  # noqa: SLF001
        1,
        "test-model",
        request.to_messages(),
        **request.to_submit_kwargs(),
    )

    private_payload = captured["prompt_payload"]
    assert private_payload["tool_choice"] == "required"
    assert private_payload["min_tool_calls"] == 1
    assert private_payload["max_tool_calls"] == 1
    assert private_payload["tool_schemas_json"] == (
        '[{"name": "lookup", "type": "function", "description": "Lookup tool", '
        '"strict": true, "parameters": {"type": "object", "properties": {}}}]'
    )
    assert private_payload["active_tool_schemas_json"] == (
        '[{"name": "lookup", "type": "object", "description": "Lookup tool", '
        '"properties": {"name": {"const": "lookup"}, "arguments": {"type": "object", '
        '"properties": {}}}, "strict": true, "required": ["name", "arguments"]}]'
    )


@pytest.mark.asyncio
async def test_gemma4_multimodal_uses_placeholder_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    model_path = tmp_path / "gemma4"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type": "gemma4"}')
    formatter = ChatFormatter(str(model_path))
    client = _make_client(formatter)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": DATA_URL},
                {"type": "input_text", "text": "What is shown?"},
            ],
        }
    ]

    captured: dict[str, Any] = {}

    def _capture_request_payload(**payload_kwargs: Any) -> bytes:
        captured["prompt_payload"] = payload_kwargs["prompts"][0]
        return b"captured"

    monkeypatch.setattr(
        client_module, "_build_request_payload", _capture_request_payload
    )

    rendered = await client.arender_prompt(
        "test-model", messages, rng_seed=1234, reasoning_effort="high"
    )
    await client._asubmit_request(  # noqa: SLF001
        1, "test-model", messages, rng_seed=1234, reasoning_effort="high"
    )

    assert formatter.image_placeholder == "<|image|>"
    assert formatter.should_clip_image_placeholder is True
    assert formatter.get_thinking_tokens() == {
        "start": "<|channel>thought\n",
        "end": "<channel|>",
    }
    assert "What is shown?" in rendered["rendered_prompt_text"]
    assert "<|image|>" not in rendered["rendered_prompt_text"]
    assert captured["prompt_payload"]["thinking_tokens"] == {
        "start": "<|channel>thought\n",
        "end": "<channel|>",
    }
    assert captured["prompt_payload"]["reasoning_effort"] == "high"
    assert any(
        segment["type"] == "image" for segment in captured["prompt_payload"]["layout"]
    )


@pytest.mark.asyncio
async def test_gemma4_multimodal_sends_thinking_delimiters_when_reasoning_disabled(
    tmp_path,
) -> None:
    model_path = tmp_path / "gemma4"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type": "gemma4"}')
    formatter = ChatFormatter(str(model_path))
    client = _make_client(formatter)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": DATA_URL},
                {"type": "input_text", "text": "What is shown?"},
            ],
        }
    ]

    rendered = await client.arender_prompt("test-model", messages, rng_seed=1234)

    assert "What is shown?" in rendered["rendered_prompt_text"]
    assert rendered["rendered_prompt_text"].endswith(
        "<|turn>model\n<|channel>thought\n<channel|>"
    )
    assert rendered["reasoning_effort"] is None
    assert rendered["thinking_tokens"] == {
        "start": "<|channel>thought\n",
        "end": "<channel|>",
    }


@pytest.mark.asyncio
async def test_gemma4_e4b_does_not_suppress_thinking_when_reasoning_disabled(
    tmp_path,
) -> None:
    model_path = tmp_path / "gemma4-e4b"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        json.dumps({"model_type": "gemma4", "text_config": {"hidden_size": 2560}})
    )
    formatter = ChatFormatter(str(model_path))
    client = _make_client(formatter)
    messages = [{"role": "user", "content": "Return only 7."}]

    rendered = await client.arender_prompt("test-model", messages, rng_seed=1234)

    assert rendered["rendered_prompt_text"].endswith("<|turn>model\n")
    assert "<|channel>thought\n<channel|>" not in rendered["rendered_prompt_text"]
    assert rendered["thinking_tokens"] == {
        "start": "<|channel>thought\n",
        "end": "<channel|>",
    }


@pytest.mark.asyncio
async def test_non_native_thinking_profile_drops_reasoning_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    model_path = tmp_path / "llama3"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type": "llama3"}')
    formatter = ChatFormatter(str(model_path))
    client = _make_client(formatter)
    messages = [{"role": "user", "content": "Return JSON."}]

    captured: dict[str, Any] = {}

    def _capture_request_payload(**payload_kwargs: Any) -> bytes:
        captured["prompt_payload"] = payload_kwargs["prompts"][0]
        return b"captured"

    monkeypatch.setattr(
        client_module, "_build_request_payload", _capture_request_payload
    )

    rendered = await client.arender_prompt(
        "test-model",
        messages,
        rng_seed=1234,
        reasoning=True,
        reasoning_effort="high",
    )
    await client._asubmit_request(  # noqa: SLF001
        1,
        "test-model",
        messages,
        rng_seed=1234,
        reasoning=True,
        reasoning_effort="high",
    )

    assert formatter.supports_native_thinking() is False
    assert formatter.get_thinking_tokens() == {
        "start": "```thinking\n",
        "end": "\n```",
    }
    assert rendered["reasoning_effort"] is None
    assert captured["prompt_payload"]["reasoning_effort"] is None
    assert captured["prompt_payload"]["thinking_tokens"] == {"start": "", "end": ""}


def test_formatter_can_use_engine_inspected_config(tmp_path) -> None:
    model_path = tmp_path / "llama.gguf"
    model_path.write_bytes(b"GGUF")

    formatter = ChatFormatter.from_config(
        str(model_path),
        {
            "model_type": "llama",
            "source_format": "gguf",
        },
    )

    assert formatter.model_type == "llama3"
    rendered = formatter.apply_template(
        [{"role": "user", "content": "hello"}],
        add_generation_prompt=True,
    )
    assert "<|start_header_id|>user<|end_header_id|>" in rendered


def test_llama3_formatter_includes_default_system_prelude(tmp_path) -> None:
    model_path = tmp_path / "llama3"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type": "llama3"}')

    formatter = ChatFormatter(str(model_path))
    rendered = formatter.apply_template([{"role": "user", "content": "hello"}])

    assert rendered.startswith(
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "Cutting Knowledge Date: December 2023\n"
        "Today Date: 26 Jul 2024\n\n"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nhello"
    )


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
        '[{"name": "lookup", "type": "function", "description": "Lookup tool", '
        '"strict": true, "parameters": {"type": "object", "properties": {}}}]'
    )
    assert private_payload["active_tool_schemas_json"] == (
        '[{"name": "lookup", "type": "object", "description": "Lookup tool", '
        '"properties": {"name": {"const": "lookup"}, "arguments": {"type": "object", '
        '"properties": {}}}, "strict": true, "required": ["name", "arguments"]}]'
    )
    assert rendered["task_name"] is None
    assert rendered["reasoning_effort"] is None
    assert private_payload["thinking_tokens"] == {
        "start": "<think>\n",
        "end": "\n</think>",
    }
