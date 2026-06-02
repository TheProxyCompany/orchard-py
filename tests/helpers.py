import json
from collections import Counter
from collections.abc import AsyncIterator

from orchard.clients.responses import ResponseEvent
from orchard.server.models.responses import (
    OutputFunctionCall,
    OutputItemAddedEvent,
    OutputItemDoneEvent,
    ResponseCompletedEvent,
)
from orchard.server.models.responses.output import OutputItem


async def drain_stream(stream: AsyncIterator[ResponseEvent]) -> dict:
    """Consume a streaming response in order, accumulating deltas.

    Returns the ordered event types, per-type counts, output items opened
    (by item type), and the accumulated reasoning / content / tool-argument
    text plus their terminal `.done` values and the function-call items.
    `field_args` reconstructs each tool argument from its `field_path`-tagged
    value deltas (untagged structural boilerplate excluded). `events` is the raw
    ordered event stream, for exact golden-snapshot comparison.
    Streams the raw deltas to stdout as they arrive (visible under ``pytest -s``).
    """
    events: list[ResponseEvent] = []
    order: list[str] = []
    counts: Counter[str] = Counter()
    items_opened: Counter[str] = Counter()
    reasoning_deltas: list[str] = []
    content_deltas: list[str] = []
    argument_deltas: list[str] = []
    field_args: dict[str, str] = {}  # field_path -> accumulated value deltas (value-only, untagged boilerplate excluded)
    output_token_ids: list[int] = []  # raw token ids, only when stream_tokens=True
    output_chunks: list[str] = []  # decoded run-text per token, joined = the raw generation
    reasoning_done: str | None = None
    content_done: str | None = None
    arguments_done: str | None = None
    function_calls: list[OutputFunctionCall] = []
    items_added: list[OutputItem] = []  # output items as opened (name known, payload still empty)
    items_done: list[OutputItem] = []   # output items as finalized (full payload + status)
    usage: dict[str, int] | None = None
    stop_token_id: int | None = None  # matched stop/EOS token id, from response.completed
    stop_token: str | None = None  # its decoded text (e.g. "<|eom_id|>")

    async for event in stream:
        events.append(event)
        order.append(event.type)
        counts[event.type] += 1
        if event.type == "response.output_item.added" and isinstance(event, OutputItemAddedEvent):
            items_opened[event.item.type] += 1
            items_added.append(event.item)
        elif event.type == "response.reasoning.delta":
            reasoning_deltas.append(event.delta)
        elif event.type == "response.reasoning.done":
            reasoning_done = event.text
        elif event.type == "response.output_token":
            # the faithful run-decoded output, streamed token by token (stream_tokens=True)
            output_token_ids.append(event.token_id)
            if event.content:
                output_chunks.append(event.content)
                print(event.content, end="", flush=True)
        elif event.type == "response.output_text.delta":
            content_deltas.append(event.delta)
        elif event.type == "response.output_text.done":
            content_done = event.text
        elif event.type == "response.function_call_arguments.delta":
            argument_deltas.append(event.delta)
            if event.field_path is not None:
                field_args[event.field_path] = field_args.get(event.field_path, "") + event.delta
        elif event.type == "response.function_call_arguments.done":
            arguments_done = event.arguments
        elif event.type == "response.output_item.done" and isinstance(event, OutputItemDoneEvent):
            items_done.append(event.item)
            if isinstance(event.item, OutputFunctionCall):
                function_calls.append(event.item)
        elif event.type == "response.completed" and isinstance(event, ResponseCompletedEvent):
            stop_token_id = event.response.stop_token_id
            stop_token = event.response.stop_token
            if stop_token:  # the decoded stop token, inline at the end of the stream
                print(stop_token, end="", flush=True)
            u = event.response.usage
            if u is not None:
                usage = {
                    "input_tokens": u.input_tokens,
                    "output_tokens": u.output_tokens,
                    "cached_tokens": u.input_tokens_details.cached_tokens if u.input_tokens_details else 0,
                }
        elif event.type == "done":
            break

    return {
        "events": events,
        "order": order,
        "counts": counts,
        "added": items_opened,
        "reasoning": "".join(reasoning_deltas),
        "reasoning_done": reasoning_done,
        "content": "".join(content_deltas),
        "content_done": content_done,
        "args": "".join(argument_deltas),
        "args_done": arguments_done,
        "field_args": field_args,
        "output_token_ids": output_token_ids,
        "generated": "".join(output_chunks),
        "stop_token_id": stop_token_id,
        "stop_token": stop_token,
        "function_calls": function_calls,
        "items_added": items_added,
        "items_done": items_done,
        "usage": usage,
    }


async def render_prompt_blue(
    client, model_id: str, *, prev_gen: str = "", **kwargs
) -> str:
    """Print, in blue, what this request's prompt appends past the prior turn.

    Renders the chat template without generating (same args as aresponses), then
    shows only the content after the prior turn's generation. `prev_gen` is what
    the model generated last turn; its tail lands verbatim in this prompt's
    history (a reasoner strips the CoT *prefix* but keeps the tail), so we find
    the longest suffix of `prev_gen` present here and show only what follows it.
    Returns the full rendered prompt.
    """
    payload = await client.arender_responses_prompt(model_id, **kwargs)
    full = payload["rendered_prompt_text"]
    new = full
    for k in range(len(prev_gen), 0, -1):
        cut = full.rfind(prev_gen[-k:])
        if cut != -1:
            new = full[cut + k:]
            break
    print(f"\033[34m{new}\033[0m", end="", flush=True)
    return full


def print_usage_summary(turns: list[dict]) -> None:
    """After the transcript, print each turn's [cached/in/out] footer in order."""
    print()
    for n, turn in enumerate(turns, start=1):
        print(f"\033[33mturn {n}\033[0m", flush=True)
        u = turn["usage"]
        if u:
            print(
                f"\033[2m[cached={u['cached_tokens']} "
                f"in={u['input_tokens']} out={u['output_tokens']}]\033[0m",
                flush=True,
            )


def parse_sse_events(raw: str) -> list[dict]:
    """Parse an SSE stream into a list of event dicts with 'event' and 'data' keys."""
    events: list[dict] = []
    current_event: str | None = None
    current_data: str = ""

    for line in raw.split("\n"):
        line = line.rstrip("\r")
        if line.startswith("event:"):
            current_event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            current_data = line[len("data:") :].strip()
        elif line == "":
            if current_data:
                if current_data == "[DONE]":
                    events.append({"event": "done", "data": "[DONE]"})
                else:
                    events.append(
                        {
                            "event": current_event,
                            "data": json.loads(current_data),
                        }
                    )
                current_event = None
                current_data = ""

    return events
