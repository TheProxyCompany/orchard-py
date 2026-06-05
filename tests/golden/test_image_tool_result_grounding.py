import base64
import json
from collections.abc import Sequence

import pytest
from golden_io import assert_or_record
from helpers import drain_stream, print_usage_summary, render_prompt_blue

from orchard.clients.client import Client, ModalArtifact
from orchard.server.models.responses import (
    OutputFunctionCall,
    OutputMessage,
    OutputStatus,
)

pytestmark = pytest.mark.asyncio

GEMMA4_MODEL = "google/gemma-4-E2B-it"
MOONDREAM3_MODEL = "moondream/moondream3-preview"
IDEOGRAM4_MODEL = "ideogram-ai/ideogram-4-fp8"

GENERATE_IMAGE = {
    "type": "function",
    "name": "generate_image",
    "description": "Generate an image from a text prompt.",
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "A concise visual prompt for the image generator.",
            }
        },
        "required": ["prompt"],
    },
}

SYSTEM = (
    "You are a multimodal assistant with image-generation tools. Use the tool "
    "when the user asks you to create an image. After the tool result is "
    "returned, inspect the image and answer from the image."
)

USER = (
    "Use generate_image to create a simple image of one red apple centered on a "
    "plain white background. After the tool returns, tell me what object is in "
    "the generated image."
)


def _image_part(artifact: ModalArtifact) -> dict[str, str]:
    encoded = base64.b64encode(artifact.data).decode("ascii")
    return {
        "type": "input_image",
        "image_url": f"data:{artifact.mime_type};base64,{encoded}",
        "detail": "auto",
    }


async def _run_gemma_generate_image_call(client: Client) -> tuple[dict, OutputFunctionCall, str]:
    conversation = [
        {"type": "message", "role": "system", "content": SYSTEM},
        {"type": "message", "role": "user", "content": USER},
    ]
    request = dict(
        input=conversation,
        core_tools=[GENERATE_IMAGE],
        tool_choice="required",
        temperature=0.0,
        deterministic=True,
        max_output_tokens=512,
        reasoning={"effort": "medium"},
        prefix_cache=False,
    )
    await render_prompt_blue(client, GEMMA4_MODEL, **request)
    stream = await client.aresponses(
        GEMMA4_MODEL, stream=True, stream_tokens=True, **request
    )
    turn = await drain_stream(stream)

    assert turn["order"][0] == "response.created"
    assert turn["order"][-1] == "done"
    assert turn["counts"]["response.created"] == 1
    assert turn["counts"]["response.in_progress"] == 1
    assert turn["counts"]["response.completed"] == 1
    assert turn["added"].get("reasoning", 0) <= 1, "generator: expected at most one reasoning block"
    if turn["added"].get("reasoning", 0):
        assert turn["counts"]["response.reasoning.done"] == 1
        assert turn["reasoning"].strip() == turn["reasoning_done"], (
            "generator: reasoning deltas != reasoning.done"
        )
    assert "response.output_text.delta" not in turn["counts"], (
        "generator: leaked message text on a tool turn"
    )
    assert turn["added"]["function_call"] == 1, "generator: expected one function_call"
    assert turn["counts"]["response.function_call_arguments.done"] == 1
    assert len(turn["function_calls"]) == 1

    opened = [item for item in turn["items_added"] if isinstance(item, OutputFunctionCall)]
    assert len(opened) == 1, "generator: expected one function_call opened"
    call = turn["function_calls"][0]
    assert opened[0].name == "generate_image"
    assert opened[0].call_id == call.call_id
    assert opened[0].arguments == "", "generator: function_call must open with empty arguments"
    assert opened[0].status == OutputStatus.IN_PROGRESS
    assert call.name == "generate_image"
    assert call.status == OutputStatus.COMPLETED

    arguments = json.loads(call.arguments)
    assert set(arguments) == {"prompt"}
    prompt = arguments["prompt"]
    assert isinstance(prompt, str) and prompt.strip()
    assert "apple" in prompt.lower(), f"generator prompt lost the requested object: {prompt!r}"

    return turn, call, prompt


async def _generate_ideogram_image(client: Client, prompt: str) -> ModalArtifact:
    artifacts = await client.images.agenerate(
        IDEOGRAM4_MODEL,
        prompt,
        height=512,
        width=512,
        num_steps=12,
        guidance_scale=7.0,
        mu=0.5,
        std=1.75,
        seed=17,
    )
    assert isinstance(artifacts, Sequence)
    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.mime_type == "image/png"
    assert artifact.data.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(artifact.data) > 1024
    return artifact


async def test_image_tool_self_loop_and_blind_verifier(client: Client):
    print(
        "\n\033[1;33m━━━ gemma4 · ideogram image tool → gemma/moondream vision ━━━\033[0m",
        flush=True,
    )

    generator, call, prompt = await _run_gemma_generate_image_call(client)
    assert_or_record("gemma4", "image_tool_self_loop", "turn1", generator["events"])
    assert_or_record("gemma4", "image_tool_blind_verifier", "generator", generator["events"])

    artifact = await _generate_ideogram_image(client, prompt)
    image = _image_part(artifact)

    conversation = [
        {"type": "message", "role": "system", "content": SYSTEM},
        {"type": "message", "role": "user", "content": USER},
        {
            "type": "function_call",
            "call_id": call.call_id,
            "name": call.name,
            "arguments": call.arguments,
        },
        {
            "type": "function_call_output",
            "call_id": call.call_id,
            "output": [image],
        },
    ]
    turn2_request = dict(
        input=conversation,
        core_tools=[GENERATE_IMAGE],
        tool_choice="none",
        temperature=0.0,
        deterministic=True,
        max_output_tokens=512,
        reasoning={"effort": "medium"},
        prefix_cache=False,
    )
    gen1 = generator["generated"] + (generator["stop_token"] or "")
    await render_prompt_blue(client, GEMMA4_MODEL, prev_gen=gen1, **turn2_request)
    stream = await client.aresponses(
        GEMMA4_MODEL, stream=True, stream_tokens=True, **turn2_request
    )
    self_loop = await drain_stream(stream)
    assert_or_record("gemma4", "image_tool_self_loop", "turn2", self_loop["events"])

    assert self_loop["order"][0] == "response.created"
    assert self_loop["order"][-1] == "done"
    assert self_loop["counts"]["response.created"] == 1
    assert self_loop["counts"]["response.in_progress"] == 1
    assert self_loop["counts"]["response.completed"] == 1
    assert self_loop["counts"].get("response.function_call_arguments.done", 0) == 0
    assert self_loop["counts"]["response.output_text.done"] == 1
    assert self_loop["content"] == self_loop["content_done"]
    msg_open = [item for item in self_loop["items_added"] if isinstance(item, OutputMessage)]
    msg_done = [item for item in self_loop["items_done"] if isinstance(item, OutputMessage)]
    assert len(msg_open) == 1 and len(msg_done) == 1
    assert msg_open[0].role == "assistant"
    assert not msg_open[0].content
    assert msg_open[0].status == OutputStatus.IN_PROGRESS
    assert msg_done[0].status == OutputStatus.COMPLETED
    assert "apple" in self_loop["content_done"].lower(), (
        f"gemma4 did not ground on the generated image: {self_loop['content_done']!r}"
    )

    verifier_request = dict(
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What is in this image? Answer with the main object."},
                    image,
                ],
            }
        ],
        temperature=0.0,
        deterministic=True,
        max_output_tokens=512,
        reasoning={"effort": "medium"},
        prefix_cache=False,
    )
    await render_prompt_blue(client, MOONDREAM3_MODEL, **verifier_request)
    stream = await client.aresponses(
        MOONDREAM3_MODEL, stream=True, stream_tokens=True, **verifier_request
    )
    verifier = await drain_stream(stream)
    print_usage_summary([generator, self_loop, verifier])
    assert_or_record("moondream3", "image_tool_blind_verifier", "verifier", verifier["events"])

    assert verifier["order"][0] == "response.created"
    assert verifier["order"][-1] == "done"
    assert verifier["counts"]["response.created"] == 1
    assert verifier["counts"]["response.in_progress"] == 1
    assert verifier["counts"]["response.completed"] == 1
    assert verifier["counts"]["response.output_text.done"] == 1
    assert verifier["content"] == verifier["content_done"]
    assert "apple" in verifier["content_done"].lower(), (
        f"moondream3 did not identify the generated image: {verifier['content_done']!r}"
    )
