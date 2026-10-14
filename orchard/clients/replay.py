"""Replaying what a model generated as the token ids it produced.

A reply the model wrote comes back on the next turn as part of the prompt. Rendered
as text and encoded again it is not always the same ids (a merge across the seam,
a special token, a split the model chose that the tokenizer would not), and one
different id ends prefix-cache reuse for the rest of the conversation. An assistant
message that carries a `generation` record is sent as those exact ids instead: the
template renders a marker in its place, and the marker becomes a `tokens` layout
segment. The next prompt is then the last prompt plus the reply, id for id.
"""

from __future__ import annotations

import json
import re
from typing import Any

# Private-use code points: no template or tokenizer gives them a meaning.
MARK_OPEN = "\ue000"
MARK_CLOSE = "\ue001"


def _is_token_ids(tokens: Any) -> bool:
    # bool is an int in Python; an id is a plain non-negative int32.
    return (
        isinstance(tokens, list)
        and bool(tokens)
        and all(type(token) is int and 0 <= token < 2**31 for token in tokens)
    )


def take_replays(
    messages: list[dict[str, Any]], model: str | None
) -> tuple[list[dict[str, Any]], list[list[int]]]:
    """Replace each assistant message's `generation` record with a marker for the
    template (`generated`) and the thinking mode it was generated in
    (`generated_thinking`), returning the ids the markers stand for. `model` is the
    model being asked, when its engine takes token segments. A record from any other
    model, or one that is not a non-empty list of token ids, is dropped and the message
    renders as text; so is every record when the conversation already contains a marker
    character, since text that looks like a marker must never be swapped for ids.
    """
    if model is not None and MARK_OPEN in json.dumps(
        messages, ensure_ascii=False, default=str
    ):
        model = None
    replays: list[list[int]] = []
    taken: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict) or "generation" not in message:
            taken.append(message)
            continue
        message = dict(message)
        generation = message.pop("generation")
        if not isinstance(generation, dict):
            generation = {}
        tokens = generation.get("tokens")
        if (
            model is not None
            and generation.get("model") == model
            and _is_token_ids(tokens)
        ):
            message["generated"] = f"{MARK_OPEN}{len(replays)}{MARK_CLOSE}"
            message["generated_thinking"] = generation.get("thinking", False)
            replays.append(tokens)
        taken.append(message)
    return taken, replays


_MARKER = re.compile(MARK_OPEN.encode() + rb"([0-9]{1,9})" + MARK_CLOSE.encode())


def splice_replays(
    prompt: str, layout: list[dict[str, Any]], replays: list[list[int]]
) -> tuple[str, list[dict[str, Any]], list[list[int]]]:
    """Cut the markers out of the prompt text and put a `tokens` segment where each
    one was. Returns the prompt, the layout, and the token segments in layout order.
    Markers are taken in order, once each; anything else stays in the text around it,
    and a request with nothing to replay comes back exactly as it went in.
    """
    if not replays:
        return prompt, layout, []
    # Layout text lengths count UTF-8 bytes.
    data = prompt.encode("utf-8")
    text = bytearray()
    spliced: list[dict[str, Any]] = []
    token_segments: list[list[int]] = []

    def push_text(part: bytes) -> None:
        if part:
            text.extend(part)
            spliced.append({"type": "text", "length": len(part)})

    cursor = 0
    for entry in layout:
        if entry.get("type") != "text":
            spliced.append(entry)
            continue
        segment = data[cursor : cursor + entry["length"]]
        cursor += entry["length"]
        run_start = 0
        for marker in _MARKER.finditer(segment):
            if int(marker.group(1)) != len(token_segments) or len(
                token_segments
            ) >= len(replays):
                continue
            push_text(segment[run_start : marker.start()])
            ids = replays[len(token_segments)]
            spliced.append({"type": "tokens", "length": len(ids)})
            token_segments.append(ids)
            run_start = marker.end()
        push_text(segment[run_start:])
    return text.decode("utf-8"), spliced, token_segments
