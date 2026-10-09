"""Agent-facing chat protocol: real structured engine events, no model/GPU required."""

import asyncio
import json

import pytest

from orchard.server.models.chat.output import ChatCompletionUsage
from orchard.server.models.chat.request import ChatMessage
from orchard.server.routes.chat import (
    gather_non_streaming_batch_response,
    stream_response_generator,
)


class Ipc:
    async def next_delta(self, queue):
        return await queue.get()


def event(index, kind, identifier, **values):
    return {
        "item_type": "tool_call",
        "output_index": index,
        "event_type": kind,
        "identifier": identifier,
        **values,
    }


def deltas():
    base = {
        "request_id": 7,
        "prompt_index": 0,
        "candidate_index": 0,
        "sequence_id": 4,
        "prompt_token_count": 40,
        "cached_token_count": 30,
    }
    return [
        {
            **base,
            "tokens": [11],
            "content": "",
            "is_final_delta": False,
            "state_events": [
                event(1, "item_started", "tool_call:write_file"),
                event(1, "content_delta", "arguments", delta='path="x", content="y"'),
            ],
        },
        {
            **base,
            "tokens": [12],
            "content": "",
            "is_final_delta": False,
            "state_events": [
                event(
                    1,
                    "item_completed",
                    "tool_call:write_file",
                    value=json.dumps(
                        {
                            "name": "write_file",
                            "arguments": {"path": "x", "content": "y"},
                        }
                    ),
                ),
                event(3, "item_started", "tool_call:bash"),
                event(
                    3,
                    "item_completed",
                    "tool_call:bash",
                    value={"name": "bash", "arguments": {"command": "cat x"}},
                ),
            ],
        },
        {
            **base,
            "tokens": [],
            "content": "",
            "is_final_delta": True,
            "finish_reason": "tool_use",
            "generation_len": 5,
            "reasoning_tokens": 2,
            "state_events": [],
        },
    ]


async def queue(values):
    result = asyncio.Queue()
    for value in values:
        await result.put(value)
    return result


@pytest.mark.asyncio
async def test_nonstream_tools_are_structured_and_keep_generation_and_cache_usage():
    response = await gather_non_streaming_batch_response(
        7, await queue(deltas()), Ipc(), [1], [1], "model", [False], released_text=True
    )
    choice = response["choices"][0]
    assert choice.finish_reason == "tool_calls"
    assert [call.function.name for call in choice.message.tool_calls] == [
        "write_file",
        "bash",
    ]
    assert json.loads(choice.message.tool_calls[0].function.arguments) == {
        "path": "x",
        "content": "y",
    }
    assert choice.message.generation == {
        "model": "model",
        "tokens": [11, 12],
        "thinking": False,
    }
    assert response["cached_tokens"] == 30
    assert len({call.id for call in choice.message.tool_calls}) == 2


@pytest.mark.asyncio
async def test_stream_emits_dense_tool_indices_then_usage_before_done():
    chunks = [
        chunk["data"]
        async for chunk in stream_response_generator(
            7,
            await queue(deltas()),
            Ipc(),
            "model",
            1,
            "model",
            [False],
            released_text=True,
            include_usage=True,
        )
    ]
    assert chunks[-1] == "[DONE]"
    parsed = [json.loads(chunk) for chunk in chunks[:-1]]
    calls = [
        call
        for chunk in parsed
        for choice in chunk["choices"]
        for call in choice["delta"].get("tool_calls", [])
    ]
    assert [call["index"] for call in calls] == [0, 1]
    assert [call["function"]["name"] for call in calls] == ["write_file", "bash"]
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "path": "x",
        "content": "y",
    }
    final = parsed[-1]
    assert final["choices"] == []
    assert final["usage"]["prompt_tokens"] == 40
    assert final["usage"]["completion_tokens"] == 5
    assert final["usage"]["prompt_tokens_details"]["cached_tokens"] == 30
    assert final["usage"]["total_tokens"] == 45
    assert parsed[-2]["choices"][0]["delta"]["generation"]["tokens"] == [11, 12]


@pytest.mark.asyncio
async def test_cut_off_tool_arguments_are_never_promoted_to_an_effect():
    values = deltas()
    values[1]["state_events"] = []
    values[-1]["finish_reason"] = "length"
    response = await gather_non_streaming_batch_response(
        7, await queue(values), Ipc(), [1], [1], "model", [False], released_text=True
    )
    assert response["choices"][0].message.tool_calls == []


def test_tool_result_ids_and_opaque_generation_roundtrip():
    tool = {
        "role": "tool",
        "content": "written",
        "tool_call_id": "call-1",
        "name": "write_file",
    }
    assert ChatMessage.model_validate(tool).model_dump() == tool
    assistant = {
        "role": "assistant",
        "content": "done",
        "generation": {"model": "gemma", "tokens": [7, 8], "thinking": True},
    }
    assert ChatMessage.model_validate(assistant).model_dump() == assistant
    usage = ChatCompletionUsage(
        input_tokens=4,
        output_tokens=2,
        reasoning_tokens=1,
        total_tokens=7,
        cached_tokens=99,
    )
    assert usage.model_dump()["prompt_tokens_details"]["cached_tokens"] == 4


def test_temperature_omission_uses_declared_default() -> None:
    from orchard.server.models.chat.request import ChatCompletionRequest

    request = ChatCompletionRequest(model="local", messages=[{"role": "user", "content": "hello"}])
    assert request.temperature == 1.0
    assert request.get_normalized_field("temperature") == [1.0]
