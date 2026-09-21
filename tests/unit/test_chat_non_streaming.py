import asyncio

import pytest

from orchard.server.exceptions import InferenceError
from orchard.server.routes.chat import gather_non_streaming_batch_response


class _IpcStub:
    """Minimal ipc_state stand-in: drains the queue with no liveness checks."""

    async def next_delta(self, queue: asyncio.Queue) -> dict:
        return await queue.get()


@pytest.mark.asyncio
async def test_non_streaming_engine_error_without_indexes_fails_immediately() -> None:
    queue = asyncio.Queue()
    await queue.put(
        {
            "request_id": 1,
            "is_final_delta": True,
            "finish_reason": "error",
            "content": "Engine process disconnected.",
            "error": "Engine process disconnected.",
        }
    )

    with pytest.raises(InferenceError, match="Engine process disconnected"):
        await asyncio.wait_for(
            gather_non_streaming_batch_response(1, queue, _IpcStub(), [1], [1]),
            timeout=0.2,
        )


def _tool_event(event_type: str, identifier: str, **fields) -> dict:
    return {
        "item_type": "tool_call",
        "event_type": event_type,
        "identifier": identifier,
        "output_index": 0,
        **fields,
    }


@pytest.mark.asyncio
async def test_tool_call_and_cached_tokens_reach_the_reply() -> None:
    """The state events are the ones the engine sent for one Qwen3.5-4B tool call."""
    queue = asyncio.Queue()
    base = {"request_id": 1, "prompt_index": 0, "candidate_index": 0, "sequence_id": 7}
    arguments = '{"city":"Paris"}'
    await queue.put(
        {
            **base,
            "prompt_token_count": 822,
            "cached_token_count": 816,
            "content": "<tool_call>",
            "tokens": [1],
        }
    )
    await queue.put(
        {
            **base,
            "content": "\n",
            "tokens": [2],
            "state_events": [_tool_event("item_started", "tool_call:get_weather")],
        }
    )
    await queue.put(
        {
            **base,
            "content": "Paris",
            "tokens": [3],
            "state_events": [
                _tool_event("item_started", "city"),
                _tool_event("content_delta", "city", delta="Paris"),
                _tool_event("item_completed", "city", value="Paris"),
            ],
        }
    )
    await queue.put(
        {
            **base,
            "content": ">",
            "tokens": [4],
            "state_events": [
                _tool_event("item_started", "arguments"),
                _tool_event("content_delta", "arguments", delta=arguments),
                _tool_event("item_completed", "arguments", value=arguments),
                _tool_event(
                    "item_completed",
                    "tool_call:get_weather",
                    value='{"name":"get_weather","arguments":{"city":"Paris"}}',
                ),
            ],
        }
    )
    await queue.put(
        {
            **base,
            "is_final_delta": True,
            "finish_reason": "stop",
            "generation_len": 25,
            "reasoning_tokens": 0,
        }
    )

    reply = await asyncio.wait_for(
        gather_non_streaming_batch_response(1, queue, _IpcStub(), [1], [1]),
        timeout=0.2,
    )

    choice = reply["choices"][0].model_dump()
    (call,) = choice["message"]["tool_calls"]
    assert call["function"] == {"name": "get_weather", "arguments": '{"city": "Paris"}'}
    assert call["id"].startswith("call_")
    assert choice["message"]["content"] == ""
    assert choice["finish_reason"] == "tool_calls"
    assert reply["cached_tokens"] == 816


def test_usage_carries_the_openai_names() -> None:
    from orchard.server.models.chat import ChatCompletionUsage

    usage = ChatCompletionUsage(
        input_tokens=120,
        output_tokens=10,
        reasoning_tokens=30,
        cached_tokens=96,
        total_tokens=160,
    ).model_dump()

    assert usage["prompt_tokens"] == 120
    assert usage["completion_tokens"] == 40
    assert usage["prompt_tokens_details"] == {"cached_tokens": 96}
    assert usage["completion_tokens_details"] == {"reasoning_tokens": 30}


def test_max_tokens_is_the_cap_under_its_older_name() -> None:
    from orchard.server.models.chat.request import ChatCompletionRequest

    def request(**cap) -> ChatCompletionRequest:
        return ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0,
            **cap,
        )

    assert request(max_tokens=300).get_normalized_field("max_completion_tokens") == [
        300
    ]
    with pytest.raises(ValueError, match="max_completion_tokens"):
        request(max_tokens=0)


def test_a_request_without_temperature_is_accepted() -> None:
    """OpenAI's `temperature` is optional; the route then samples with the model
    profile's defaults (`"temperature" not in request.model_fields_set`)."""
    from orchard.server.models.chat.request import ChatCompletionRequest

    request = ChatCompletionRequest(
        model="m", messages=[{"role": "user", "content": "hi"}]
    )

    assert "temperature" not in request.model_fields_set
