from __future__ import annotations

import json
from typing import Any, ClassVar

import pytest

import orchard.clients.client as client_module
from orchard.app.model_registry import ModelInfo
from orchard.clients.client import Client
from orchard.clients.responses import ResponsesRequest
from orchard.engine import ClientDelta
from orchard.formatter.formatter import ChatFormatter, determine_model_type

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
    end_of_sequence = ""
    roles = _FakeRoles()


class _FakeFormatter:
    control_tokens = _FakeControlTokens()
    image_placeholder = "<|image|>"
    should_clip_image_placeholder = True
    capabilities: ClassVar[dict[str, Any]] = {"thinking": {"native": True}}
    generation_defaults: ClassVar[dict[str, Any]] = {}

    def apply_template(
        self,
        messages: list[dict[str, Any]],
        *,
        reasoning: bool = False,
        reasoning_effort: str | None = None,
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

    def get_output_frame_tokens(self) -> dict[str, str]:
        return {}

    def get_thinking_tokens(self) -> dict[str, str]:
        return {"start": "<think>\n", "end": "\n</think>"}

    def supports_native_thinking(self) -> bool:
        return True

    def get_generation_defaults(self, profile: str = "default") -> dict[str, Any]:
        return dict(self.generation_defaults)

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


@pytest.mark.parametrize(
    ("source_type", "profile_type"),
    [
        ("llama", "llama3"),
        ("llama3", "llama3"),
        ("moondream", "moondream3"),
        ("moondream3", "moondream3"),
        ("gemma4", "gemma4"),
        ("gemma4_text", "gemma4"),
        ("qwen3_5", "qwen3_5"),
        ("qwen3_5_text", "qwen3_5"),
        ("qwen3_5_moe", "qwen3_5"),
        ("lfm2", "lfm2"),
        ("lfm2_moe", "lfm2_moe"),
        ("afmoe", "afmoe"),
        ("glm4_moe", "glm4_moe"),
        ("gpt_oss", "gpt_oss"),
        ("granite_switch", "granite_switch"),
        ("nemotron_h", "nemotron_h"),
        ("olmo_hybrid", "olmo_hybrid"),
        ("openai_privacy_filter", "openai_privacy_filter"),
        ("phi3", "phi3"),
    ],
)
def test_determine_model_type_maps_known_profiles(source_type: str, profile_type: str) -> None:
    assert determine_model_type({"model_type": source_type}) == profile_type


def test_determine_model_type_requires_model_type() -> None:
    with pytest.raises(ValueError, match="model_type"):
        determine_model_type({})


@pytest.mark.parametrize(
    ("source_type", "profile_type", "expected_prefix"),
    [
        ("lfm2", "lfm2", "<|im_start|>assistant\n"),
        ("lfm2_moe", "lfm2_moe", "<|im_start|>assistant\n"),
        ("phi3", "phi3", "<|im_start|>assistant<|im_sep|>"),
        ("olmo_hybrid", "olmo_hybrid", "<|im_start|>assistant\n"),
        ("nemotron_h", "nemotron_h", "<|im_start|>assistant\n<think>\n"),
        ("granite_switch", "granite_switch", "<|start_of_role|>assistant<|end_of_role|>"),
    ],
)
def test_new_text_model_profiles_render_generation_prompt(
    tmp_path, source_type: str, profile_type: str, expected_prefix: str
) -> None:
    model_path = tmp_path / source_type
    model_path.mkdir()
    (model_path / "config.json").write_text(json.dumps({"model_type": source_type}))

    formatter = ChatFormatter(str(model_path))
    rendered = formatter.apply_template(
        [{"role": "user", "content": "hello"}],
        reasoning=source_type == "nemotron_h",
    )

    assert formatter.model_type == profile_type
    assert rendered.endswith(expected_prefix)


def test_afmoe_trinity_formatter_renders_reasoning_and_json_tool_calls(
    tmp_path,
) -> None:
    model_path = tmp_path / "trinity-mini"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type": "afmoe"}')

    formatter = ChatFormatter(str(model_path))
    rendered = formatter.apply_template(
        [
            {"role": "system", "content": "Follow the test instruction."},
            {"role": "user", "content": "Use lookup."},
            {
                "role": "assistant",
                "content": "<think>\nNeed lookup.\n</think>\nCalling lookup.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": {"query": "orchard"},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"result":"ok"}',
            },
            {"role": "user", "content": "Continue."},
        ],
        reasoning=True,
        tools=[
            {
                "name": "lookup",
                "description": "Lookup a value.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ],
    )

    assert formatter.model_type == "afmoe"
    assert formatter.get_thinking_tokens() == {
        "start": "<think>\n",
        "end": "\n</think>",
    }
    assert formatter.capabilities["thinking"]["default"] is True
    assert formatter.get_tool_calling_tokens()["formats"][0] == {
        "name": "json",
        "call_start": "<tool_call>\n",
        "inline_start": "",
        "channel": "",
        "recipient_prefix": "",
        "constraint_prefix": "",
        "constraint": "",
        "message": "",
        "call_end": "\n</tool_call>",
    }
    assert rendered.startswith("<|im_start|>system\n# Tools")
    assert "Follow the test instruction." in rendered
    assert "<|im_start|>user\nUse lookup.<|im_end|>\n" in rendered
    assert "<|im_start|>assistant\nCalling lookup.\n<tool_call>\n" in rendered
    assert '"name":"lookup"' in rendered
    assert '"arguments":{"query": "orchard"}' in rendered
    assert "<function=" not in rendered
    assert "<parameter=" not in rendered
    assert '<tool_response>\n{"result":"ok"}\n</tool_response>' in rendered
    assert rendered.endswith("<|im_start|>assistant\n<think>\n")

    non_reasoning = formatter.apply_template(
        [{"role": "user", "content": "Say hi."}],
        reasoning=False,
    )
    assert non_reasoning.endswith("<|im_start|>assistant\n")


def test_granite_switch_profile_inserts_adapter_tokens(tmp_path) -> None:
    model_path = tmp_path / "granite-switch"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type": "granite_switch"}')

    formatter = ChatFormatter(str(model_path))

    lora = formatter.apply_template(
        [{"role": "user", "content": "cite this"}],
        task="citations",
    )
    assert lora.startswith("<|citations|><|start_of_role|>user<|end_of_role|>")

    alora = formatter.apply_template(
        [{"role": "user", "content": "rewrite this"}],
        task="query_rewrite",
    )
    assert alora.endswith(
        "<|query_rewrite|><|start_of_role|>assistant<|end_of_role|>"
    )


def test_phi_profile_preserves_system_messages_and_gates_thinking(tmp_path) -> None:
    model_path = tmp_path / "phi4"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type": "phi3"}')

    formatter = ChatFormatter(str(model_path))
    rendered = formatter.apply_template(
        [
            {"role": "system", "content": "End every response with 7-4-7."},
            {"role": "user", "content": "What is your name?"},
        ],
        reasoning=False,
    )

    assert "You are Phi, a language model trained by Microsoft" in rendered
    assert "End every response with 7-4-7." in rendered
    assert "Structure responses with <think>" not in rendered
    assert "<|im_start|>user<|im_sep|>What is your name?<|im_end|>" in rendered

    reasoning_rendered = formatter.apply_template(
        [{"role": "user", "content": "Think."}],
        reasoning=True,
    )
    assert "Structure responses with <think>" in reasoning_rendered


@pytest.mark.asyncio
async def test_afmoe_trinity_defaults_to_native_reasoning(tmp_path) -> None:
    model_path = tmp_path / "trinity-mini"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type": "afmoe"}')

    client = _make_client(ChatFormatter(str(model_path)))
    rendered = await client.arender_prompt(
        "test-model",
        [{"role": "user", "content": "Say hi."}],
    )

    assert rendered["rendered_prompt_text"].endswith("<|im_start|>assistant\n<think>\n")
    assert rendered["reasoning_effort"] == "medium"


def test_glm4_moe_intellect_formatter_renders_reasoning_and_xml_tool_calls(
    tmp_path,
) -> None:
    model_path = tmp_path / "intellect-31"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type": "glm4_moe"}')

    formatter = ChatFormatter(str(model_path))
    rendered = formatter.apply_template(
        [
            {"role": "system", "content": "Follow the test instruction."},
            {"role": "user", "content": "Use lookup."},
            {
                "role": "assistant",
                "reasoning_content": "Need lookup.",
                "content": "Calling lookup.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": {"query": "orchard"},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"result":"ok"}',
            },
            {"role": "user", "content": "Continue."},
        ],
        reasoning=True,
        tools=[
            {
                "name": "lookup",
                "description": "Lookup a value.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ],
    )

    assert formatter.model_type == "glm4_moe"
    assert formatter.get_thinking_tokens() == {
        "start": "<think>",
        "end": "</think>",
    }
    assert formatter.capabilities["thinking"]["default"] is True
    assert formatter.get_tool_calling_tokens()["formats"][0] == {
        "name": "xml",
        "call_start": "<tool_call>\n",
        "inline_start": "",
        "channel": "",
        "recipient_prefix": "",
        "constraint_prefix": "",
        "constraint": "",
        "message": "",
        "call_end": "\n</tool_call>",
    }
    assert rendered.startswith("<|im_start|>system\nFollow the test instruction.")
    assert "<function>\n<name>lookup</name>" in rendered
    assert "<|im_start|>user\nUse lookup.<|im_end|>\n" in rendered
    assert "<think>Need lookup.</think>" in rendered
    assert "<tool_call>\n<function=lookup>\n" in rendered
    assert "<parameter=query>\norchard\n</parameter>" in rendered
    assert '<tool_response>\n{"result":"ok"}\n</tool_response>' in rendered
    assert rendered.endswith("<|im_start|>assistant\n<think>")


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
    assert formatter.get_output_frame_tokens() == {
        "marker.channel": "<|channel>",
        "marker.message": "\n",
        "channel.analysis": "thought",
    }
    assert "What is shown?" in rendered["rendered_prompt_text"]
    assert "<|image|>" not in rendered["rendered_prompt_text"]
    assert captured["prompt_payload"]["thinking_tokens"] == {
        "start": "<|channel>thought\n",
        "end": "<channel|>",
    }
    assert captured["prompt_payload"]["output_frame_tokens"] == {
        "marker.channel": "<|channel>",
        "marker.message": "\n",
        "channel.analysis": "thought",
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
async def test_gemma4_tool_turn_preserves_assistant_reasoning_history(
    tmp_path,
) -> None:
    model_path = tmp_path / "gemma4-moe"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma4",
                "text_config": {"hidden_size": 2816, "num_experts": 128},
            }
        )
    )
    formatter = ChatFormatter(str(model_path))
    client = _make_client(formatter)
    messages = [
        {"role": "user", "content": "Use the schedule tool for Tuesday."},
        {
            "role": "assistant",
            "content": "",
            "reasoning": "The schedule tool is the correct tool.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup_schedule",
                        "arguments": {"day": "Tuesday"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": '{"status":"ok"}',
        },
    ]

    rendered = await client.arender_prompt(
        "test-model", messages, rng_seed=1234, reasoning=True
    )

    prompt = rendered["rendered_prompt_text"]
    assert rendered["reasoning_effort"] == "medium"
    assert "<|turn>agent" not in prompt
    assert (
        "<|turn>model\n<|channel>thought\n"
        "The schedule tool is the correct tool.\n<channel|>"
    ) in prompt
    assert (
        '<|tool_call>call:lookup_schedule{day:<|"|>Tuesday<|"|>}<tool_call|>' in prompt
    )
    assert (
        '<|tool_response>response:lookup_schedule{value:<|"|>{"status":"ok"}<|"|>}<tool_response|>'
        in prompt
    )


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


def test_gpt_oss_formatter_renders_harmony_prompt(tmp_path) -> None:
    model_path = tmp_path / "gpt-oss"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type": "gpt_oss"}')

    formatter = ChatFormatter(str(model_path))
    rendered = formatter.apply_template([{"role": "user", "content": "hello"}])

    assert formatter.model_type == "gpt_oss"
    assert formatter.supports_native_thinking() is True
    assert formatter.get_thinking_tokens() == {
        "start": "<|channel|>analysis<|message|>",
        "end": "<|end|>",
    }
    assert rendered.startswith(
        "<|start|>system<|message|>"
        "You are ChatGPT, a large language model trained by OpenAI.\n"
        "Knowledge cutoff: 2024-06\n\n"
        "Reasoning: medium\n\n"
        "# Valid channels: analysis, commentary, final."
    )
    assert "<|start|>user<|message|>hello<|end|><|start|>assistant" in rendered


def test_gpt_oss_formatter_preserves_system_and_developer_messages(tmp_path) -> None:
    model_path = tmp_path / "gpt-oss"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type": "gpt_oss"}')

    formatter = ChatFormatter(str(model_path))
    rendered = formatter.apply_template(
        [
            {"role": "system", "content": "System rule."},
            {"role": "developer", "content": "Developer rule."},
            {"role": "user", "content": "hello"},
        ]
    )

    assert (
        "<|start|>developer<|message|># Instructions\n\nSystem rule.<|end|>" in rendered
    )
    assert (
        "<|start|>developer<|message|># Instructions\n\nDeveloper rule.<|end|>"
        in rendered
    )
    assert "<|start|>user<|message|>hello<|end|><|start|>assistant" in rendered


def test_gpt_oss_formatter_renders_harmony_tool_history(tmp_path) -> None:
    model_path = tmp_path / "gpt-oss"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type": "gpt_oss"}')

    formatter = ChatFormatter(str(model_path))
    rendered = formatter.apply_template(
        [
            {"role": "user", "content": "Use lookup."},
            {
                "role": "assistant",
                "content": "Need lookup.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"query":"orchard"}',
                        },
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "rank",
                            "arguments": {"target": "orchard"},
                        },
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_2",
                "content": '{"score":1}',
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"result":"ok"}',
            },
        ],
        reasoning_effort="high",
        tools=[
            {
                "type": "function",
                "name": "lookup",
                "description": "Lookup a value.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"],
                },
            },
            {
                "type": "function",
                "name": "rank",
                "description": "Rank a value.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "Target"}
                    },
                    "required": ["target"],
                },
            },
        ],
    )

    assert "namespace functions" in rendered
    assert "Reasoning: high" in rendered
    assert (
        "<|start|>assistant<|channel|>analysis<|message|>Need lookup.<|end|>"
        in rendered
    )
    assert (
        '<|start|>assistant<|channel|>commentary to=functions.lookup <|constrain|>json<|message|>{"query":"orchard"}<|call|>'
        in rendered
    )
    assert (
        '<|start|>assistant<|channel|>commentary to=functions.rank <|constrain|>json<|message|>{"target": "orchard"}<|call|>'
        in rendered
    )
    assert (
        '<|start|>functions.lookup to=assistant<|channel|>commentary<|message|>{"result":"ok"}<|end|>'
        in rendered
    )
    assert (
        '<|start|>functions.rank to=assistant<|channel|>commentary<|message|>{"score":1}<|end|>'
        in rendered
    )
    assert formatter.get_tool_calling_tokens()["formats"][0] == {
        "name": "harmony",
        "call_start": "<|start|>assistant",
        "inline_start": "",
        "channel": "commentary",
        "recipient_prefix": " to=functions.",
        "constraint_prefix": " ",
        "constraint": "json",
        "message": "<|message|>",
        "call_end": "<|call|>",
    }
    assert formatter.get_output_frame_tokens() == {
        "marker.start": "<|start|>",
        "marker.channel": "<|channel|>",
        "marker.message": "<|message|>",
        "marker.constrain": "<|constrain|>",
        "marker.call": "<|call|>",
        "marker.end": "<|end|>",
        "marker.return": "<|return|>",
        "channel.analysis": "analysis",
        "channel.commentary": "commentary",
        "channel.final": "final",
    }


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


@pytest.mark.asyncio
async def test_arender_prompt_uses_profile_generation_defaults() -> None:
    formatter = _FakeFormatter()
    formatter.generation_defaults = {
        "temperature": 0.6,
        "top_p": 0.9,
        "top_k": 20,
        "min_p": 0.05,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.1,
    }
    client = _make_client(formatter)

    rendered = await client.arender_prompt(
        "test-model",
        [{"role": "user", "content": "Hello"}],
    )

    assert rendered["sampling_params"]["temperature"] == 0.6
    assert rendered["sampling_params"]["top_p"] == 0.9
    assert rendered["sampling_params"]["top_k"] == 20
    assert rendered["sampling_params"]["min_p"] == 0.05
    assert rendered["sampling_params"]["presence_penalty"] == 1.5
    assert rendered["sampling_params"]["repetition_penalty"] == 1.1


@pytest.mark.asyncio
async def test_arender_prompt_explicit_sampling_overrides_profile_defaults() -> None:
    formatter = _FakeFormatter()
    formatter.generation_defaults = {
        "temperature": 0.6,
        "top_p": 0.9,
        "top_k": 20,
    }
    client = _make_client(formatter)

    rendered = await client.arender_prompt(
        "test-model",
        [{"role": "user", "content": "Hello"}],
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
    )

    assert rendered["sampling_params"]["temperature"] == 0.0
    assert rendered["sampling_params"]["top_p"] == 1.0
    assert rendered["sampling_params"]["top_k"] == -1
