"""What the engine receives around an image or an audio clip, checked under Jinja2.

A Hugging Face processor adds tokens around the soft tokens it expands a placeholder
into; Orchard has no processor step, so the chat profile writes them as text. These
tests cut the prompt along the layout, the way the engine does, and compare every
piece with a literal. The same literals are checked under minijinja in orchard-rs
(src/formatter/mod.rs).
"""

from __future__ import annotations

from typing import Any

from orchard.formatter.formatter import ChatFormatter
from orchard.formatter.multimodal import (
    build_multimodal_layout,
    build_multimodal_messages,
)

GEMMA4_E2B = {"model_type": "gemma4", "text_config": {"hidden_size": 1536}}
IMAGE = {"type": "input_image", "image_url": "data:image/png;base64,AA=="}
AUDIO = {"type": "input_audio", "data": [0.0, 0.5, -0.5]}


def _text(value: str) -> dict[str, Any]:
    return {"type": "input_text", "text": value}


def _segments(config: dict[str, Any], items: list[dict[str, Any]]) -> list[str]:
    """Each text segment as the bytes the engine tokenizes on their own, each image
    or audio segment as `[image]` or `[audio]`."""
    formatter = ChatFormatter.from_config("/tmp/model", config)
    messages, images, audio, capabilities, order = build_multimodal_messages(
        formatter=formatter, items=items
    )
    rendered = formatter.apply_template(messages, reasoning=False)
    layout = build_multimodal_layout(
        rendered,
        images,
        audio,
        capabilities,
        order,
        formatter.image_placeholder,
        formatter.should_clip_image_placeholder,
        audio_placeholder=formatter.get_audio_placeholder(),
        coord_placeholder=formatter.get_coord_placeholder(),
    )
    prompt = formatter.strip_template_placeholders(rendered).encode()
    segments, cursor = [], 0
    for segment in layout:
        if segment["type"] == "text":
            segments.append(prompt[cursor : cursor + segment["length"]].decode())
            cursor += segment["length"]
        else:
            segments.append(f"[{segment['type']}]")
    assert cursor == len(prompt), "the text segments must cover the prompt exactly"
    return segments


def test_gemma4_wraps_every_image_in_its_begin_and_end_tokens() -> None:
    """The reference is `<|image>`, the soft tokens, `<image|>` (token ids 255999,
    258880 x N, 258882) with no newline on either side. The engine supplies the soft
    tokens, so the delimiters are the last text token before the image and the first
    one after it."""
    one_image = [
        {
            "role": "user",
            "content": [
                _text("Here is a picture."),
                IMAGE,
                _text("In one sentence, what does it show?"),
            ],
        }
    ]
    assert _segments(GEMMA4_E2B, one_image) == [
        "<bos><|turn>user\nHere is a picture.<|image>",
        "[image]",
        "<image|>In one sentence, what does it show?<turn|>\n<|turn>model\n",
    ]

    two_images = [
        {
            "role": "user",
            "content": [
                _text("Here is the first picture."),
                IMAGE,
                _text("Here is the second picture."),
                IMAGE,
                _text("What do the two pictures have in common?"),
            ],
        }
    ]
    assert _segments(GEMMA4_E2B, two_images) == [
        "<bos><|turn>user\nHere is the first picture.<|image>",
        "[image]",
        "<image|>Here is the second picture.<|image>",
        "[image]",
        "<image|>What do the two pictures have in common?<turn|>\n<|turn>model\n",
    ]

    # Two images with nothing between them still get a closing and an opening token.
    adjacent = [{"role": "user", "content": [IMAGE, IMAGE, _text("Compare them.")]}]
    assert _segments(GEMMA4_E2B, adjacent) == [
        "<bos><|turn>user\n<|image>",
        "[image]",
        "<image|><|image>",
        "[image]",
        "<image|>Compare them.<turn|>\n<|turn>model\n",
    ]


def test_gemma4_wraps_audio_in_its_begin_and_end_tokens() -> None:
    """`<|audio>` and `<audio|>` (256000 and 258883). The `<|audio|>` placeholder itself
    must not reach the engine as text: the engine refuses a prompt whose text is longer
    than its layout says."""
    listen = [
        {
            "role": "user",
            "content": [_text("Listen."), AUDIO, _text("What do you hear?")],
        }
    ]
    assert _segments(GEMMA4_E2B, listen) == [
        "<bos><|turn>user\nListen.<|audio>",
        "[audio]",
        "<audio|>What do you hear?<turn|>\n<|turn>model\n",
    ]


def test_gemma4_puts_a_tool_result_image_after_the_response_block() -> None:
    """Where the Hugging Face template and processor put it; the block keeps the text."""
    conversation = [
        {"role": "user", "content": "Create an image of an apple."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_image",
                    "type": "function",
                    "function": {
                        "name": "generate_image",
                        "arguments": {"prompt": "a red apple"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_image",
            "content": [{"type": "text", "text": "done"}, IMAGE],
        },
    ]
    up_to_the_image = (
        "<bos><|turn>user\nCreate an image of an apple.<turn|>\n<|turn>model\n"
        '<|tool_call>call:generate_image{prompt:<|"|>a red apple<|"|>}<tool_call|>'
        '<|tool_response>response:generate_image{value:<|"|>done<|"|>}<tool_response|>'
        "<|image>"
    )
    assert _segments(GEMMA4_E2B, conversation) == [
        up_to_the_image,
        "[image]",
        "<image|>",
    ]


def test_gemma3_sets_an_image_apart_with_blank_lines() -> None:
    """Gemma 3's delimiters were in its template already. What the Hugging Face
    processor adds besides the soft tokens is a blank line on each side:
    "\\n\\n<start_of_image>", the soft tokens, "<end_of_image>\\n\\n"."""
    items = [
        {
            "role": "user",
            "content": [
                _text("Here is a picture."),
                IMAGE,
                _text("What does it show?"),
            ],
        }
    ]
    before, image, after = _segments({"model_type": "gemma3"}, items)
    assert before.endswith("user\nHere is a picture.\n\n<start_of_image>"), before
    assert image == "[image]"
    assert after == (
        "<end_of_image>\n\nWhat does it show?<end_of_turn>\n<start_of_turn>model\n"
    )
