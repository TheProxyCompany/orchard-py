from typing import Any

from orchard.clients.replay import MARK_CLOSE, MARK_OPEN, splice_replays, take_replays


def _assistant(generation: dict[str, Any]) -> dict[str, Any]:
    return {"role": "assistant", "content": "hi", "generation": generation}


def _text_layout(prompt: str) -> list[dict[str, Any]]:
    return [{"type": "text", "length": len(prompt.encode("utf-8"))}]


def test_a_reply_from_the_same_model_becomes_a_token_segment() -> None:
    history = [_assistant({"model": "m", "tokens": [7, 8, 9], "thinking": True})]
    messages, replays = take_replays(history, "m")
    assert replays == [[7, 8, 9]]
    assert messages[0]["generated_thinking"] is True
    assert "generation" not in messages[0]
    assert "generation" in history[0]

    prompt = f"<a>{messages[0]['generated']}<end><b>"
    text, layout, token_segments = splice_replays(prompt, _text_layout(prompt), replays)
    assert text == "<a><end><b>"
    assert [(entry["type"], entry["length"]) for entry in layout] == [
        ("text", 3),
        ("tokens", 3),
        ("text", 8),
    ]
    assert token_segments == replays


def test_a_reply_from_another_model_or_an_older_engine_stays_text() -> None:
    history = [_assistant({"model": "other", "tokens": [7]})]
    for model in ("m", None):
        messages, replays = take_replays(history, model)
        assert replays == []
        assert "generated" not in messages[0]
        assert "generation" not in messages[0]


def test_a_record_that_is_not_a_clean_list_of_token_ids_stays_text() -> None:
    for tokens in ([], [7, -1], [7, 2**31], [7, "8"], [7, 8.5], [7, True], "7", None):
        messages, replays = take_replays(
            [_assistant({"model": "m", "tokens": tokens})], "m"
        )
        assert replays == [], tokens
        assert "generated" not in messages[0], tokens


def test_a_marker_character_in_the_conversation_turns_replay_off() -> None:
    typed = _assistant({"model": "m", "tokens": [7]})
    typed["content"] = [{"type": "text", "text": f"x{MARK_OPEN}0{MARK_CLOSE}"}]
    messages, replays = take_replays([typed], "m")
    assert replays == []
    assert "generated" not in messages[0]


def test_only_the_next_marker_in_order_is_spliced_and_nothing_else_moves() -> None:
    # Nothing to replay: the request is untouched, stray marker characters included.
    stray = f"a{MARK_OPEN}9{MARK_CLOSE}b{MARK_OPEN}c"
    layout = _text_layout(stray)
    assert splice_replays(stray, layout, []) == (stray, layout, [])
    # Out of order, repeated, unterminated and over-long markers stay inside the text.
    prompt = (
        f"{MARK_OPEN}1{MARK_CLOSE}a{MARK_OPEN}0{MARK_CLOSE}b{MARK_OPEN}0{MARK_CLOSE}"
        f"{MARK_OPEN}{'9' * 5000}{MARK_CLOSE}{MARK_OPEN}1"
    )
    text, spliced, token_segments = splice_replays(
        prompt, _text_layout(prompt), [[1], [2]]
    )
    assert token_segments == [[1]]
    assert [entry["type"] for entry in spliced] == ["text", "tokens", "text"]
    assert len(text) + len(f"{MARK_OPEN}0{MARK_CLOSE}") == len(prompt)


def test_many_stray_marker_characters_are_handled_in_linear_time() -> None:
    import time

    prompt = MARK_OPEN * 200_000
    started = time.perf_counter()
    text, spliced, token_segments = splice_replays(prompt, _text_layout(prompt), [[1]])
    assert time.perf_counter() - started < 1.0
    assert (text, len(spliced), token_segments) == (prompt, 1, [])


def test_text_lengths_count_bytes_and_other_segments_keep_their_place() -> None:
    messages, replays = take_replays(
        [_assistant({"model": "m", "tokens": [5, 6], "thinking": False})], "m"
    )
    before, after = "héllo ", " wörld"
    prompt = before + messages[0]["generated"] + after
    image = {"type": "image", "length": 4}
    layout = [
        {"type": "text", "length": len(before.encode("utf-8"))},
        image,
        {
            "type": "text",
            "length": len((messages[0]["generated"] + after).encode("utf-8")),
        },
    ]
    text, layout, token_segments = splice_replays(prompt, layout, replays)
    assert text == before + after
    assert layout == [
        {"type": "text", "length": len(before.encode("utf-8"))},
        image,
        {"type": "tokens", "length": 2},
        {"type": "text", "length": len(after.encode("utf-8"))},
    ]
    assert token_segments == [[5, 6]]
