"""How every chat profile renders earlier assistant turns, checked under Jinja2.

The same two tests run under minijinja in orchard-rs (src/formatter/mod.rs).
"""

from __future__ import annotations

from typing import Any

import pytest

from orchard.formatter.formatter import ChatFormatter

PROFILES = [
    "llama",
    "gemma3",
    "gemma4",
    "gemma4u",
    "qwen2",
    "qwen3_5",
    "lfm2",
    "lfm2_moe",
    "olmo_hybrid",
    "nemotron_h",
    "granite_switch",
    "gpt_oss",
    "proxy_i",
    "afmoe",
    "glm4_moe",
    "laguna",
]
MARKER = "\ue0000\ue001"
# The client renames `assistant` to `agent` before rendering; both must work.
ROLES = ["assistant", "agent"]


def _formatter(model_type: str) -> ChatFormatter:
    return ChatFormatter.from_config(f"/tmp/{model_type}", {"model_type": model_type})


def _message(role: str, content: str, **fields: Any) -> dict[str, Any]:
    return {"role": role, "content": content, **fields}


def test_qwen2_uses_native_chatml_and_hermes_tool_history() -> None:
    formatter = _formatter("qwen2")
    assert not formatter.supports_native_thinking()
    assert formatter.get_generation_defaults()["repetition_penalty"] == 1.05
    assert formatter.apply_template([_message("user", "Hello")]) == (
        "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. "
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
    )
    history = [
        _message("system", "Be precise."),
        _message("user", "Find it."),
        _message(
            "agent",
            "",
            tool_calls=[
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"q":"São Paulo","limit":2}',
                    },
                }
            ],
        ),
        _message("tool", "first"),
        _message("tool", "second"),
    ]
    assert formatter.apply_template(history) == (
        "<|im_start|>system\nBe precise.<|im_end|>\n"
        "<|im_start|>user\nFind it.<|im_end|>\n"
        '<|im_start|>assistant\n<tool_call>\n{"name": "lookup", "arguments": '
        '{"q": "São Paulo", "limit": 2}}\n</tool_call><|im_end|>\n'
        "<|im_start|>user\n<tool_response>\nfirst\n</tool_response>"
        "\n<tool_response>\nsecond\n</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n"
    )


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("next_thinking", [False, True])
@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("model_type", PROFILES)
def test_a_replayed_reply_extends_the_prompt_it_answered(
    model_type: str, thinking: bool, next_thinking: bool, role: str
) -> None:
    """A reply the model generated comes back as a `generated` marker. Every chat
    profile must render the conversation so far exactly as it was when that reply was
    generated, then the marker, then the rest: only then is the next prompt the last
    prompt plus the reply, which is what lets the engine reuse all of its cache.
    """
    formatter = _formatter(model_type)
    asked = [_message("system", "Be brief."), _message("user", "first question")]
    first = formatter.apply_template(asked, reasoning=thinking)

    reply = _message(
        role,
        "ignored: the marker stands for the reply",
        generated=MARKER,
        generated_thinking=thinking,
    )

    # Some models switch thinking at the head of the prompt (Gemma 4 writes `<|think|>`
    # into the system turn), so the conversation re-renders in the next request's mode.
    # The generation prompt the reply answered stays as it was asked.
    def conversation(mode: bool) -> str:
        return formatter.apply_template(
            asked, add_generation_prompt=False, reasoning=mode
        )

    assert first.startswith(conversation(thinking))
    asked_prompt = first[len(conversation(thinking)) :]
    second = formatter.apply_template(
        [*asked, reply, _message("user", "second question")], reasoning=next_thinking
    )

    assert second.startswith(conversation(next_thinking) + asked_prompt + MARKER)


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("model_type", PROFILES)
def test_reasoning_stays_in_the_conversation_and_earlier_turns_do_not_move(
    model_type: str, role: str
) -> None:
    """A reply that arrives as text (another model wrote it, or an HTTP client sent
    it) keeps its reasoning on every turn, and adding later messages never changes how
    an earlier turn renders: the conversation so far stays a prefix of the conversation.
    """
    formatter = _formatter(model_type)
    thinking = formatter.supports_native_thinking()
    two_turns = [
        _message("system", "Be brief."),
        _message("user", "first question"),
        _message(role, "first answer", reasoning_content="FIRST-REASONING"),
        _message("user", "second question"),
    ]
    three_turns = [
        *two_turns,
        _message(role, "second answer", reasoning_content="SECOND-REASONING"),
        _message("user", "third question"),
    ]

    # Rendered without a generation prompt, the shorter conversation is where the
    # longer one starts.
    shorter = formatter.apply_template(
        two_turns, add_generation_prompt=False, reasoning=thinking
    )
    longer = formatter.apply_template(
        three_turns, add_generation_prompt=False, reasoning=thinking
    )

    assert longer.startswith(shorter), (
        "an earlier turn renders differently once later messages exist"
    )
    if thinking:
        assert "FIRST-REASONING" in longer, "reasoning dropped from an earlier turn"
        assert "SECOND-REASONING" in longer, "reasoning dropped from an earlier turn"
