# Orchard

[![PyPI](https://img.shields.io/pypi/v/orchard.svg)](https://pypi.org/project/orchard/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![macOS](https://img.shields.io/badge/macOS-14%2B-111111.svg)](#requirements)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-required-024645.svg)](#requirements)

Standalone local inference for Apple Silicon, from Python.

`orchard` is the standalone Python package for Orchard. Install it into a
Python environment, call the embedded client directly from scripts or services,
or start the optional OpenAI-compatible HTTP server when another process needs
to talk to local models. It wraps the Proxy Inference Engine, a local C++ and
Metal runtime built for streaming, continuous batching, multiple loaded models,
structured output, tool calls, and multimodal inputs.

`macOS 14+` | `Apple Silicon` | `Python 3.12+` | `Apache-2.0`

[Official docs](https://docs.theproxycompany.com/orchard/) | Quickstart | Client |
Streaming | Responses | Server | Batching | Multimodal | Structured Output |
Tool Use | Models

## Install

```bash
uv venv
source .venv/bin/activate
uv pip install orchard
```

If you are not using `uv`, install inside a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install orchard
```

The first request downloads the Orchard engine binary and the model weights you
ask for. The engine binary is cached under `~/.orchard/`; Hugging Face model
files use the normal Hugging Face cache.

## Quickstart

Use the Python client directly when you are writing a Python app, notebook,
worker, or evaluation job. You do not need to start the HTTP server for this
path.

Create `hello_orchard.py`:

```python
from orchard.engine.inference_engine import InferenceEngine

MODEL = "google/gemma-4-E2B-it"

with InferenceEngine(load_models=[MODEL]) as engine:
    client = engine.client()
    response = client.chat(
        MODEL,
        [{"role": "user", "content": "Write one sentence about local AI."}],
        temperature=0.0,
        max_generated_tokens=64,
    )
    print(response.text)
```

Run it:

```bash
python hello_orchard.py
```

For larger Macs, try `google/gemma-4-E4B-it`,
`meta-llama/Llama-3.1-8B-Instruct`, or `Qwen/Qwen3.5-4B`.

## Streaming

`client.chat(..., stream=True)` returns token deltas as the engine produces
them.

```python
from orchard.engine.inference_engine import InferenceEngine

MODEL = "google/gemma-4-E2B-it"

with InferenceEngine(load_models=[MODEL]) as engine:
    client = engine.client()
    stream = client.chat(
        MODEL,
        [{"role": "user", "content": "Count from one to five."}],
        stream=True,
        temperature=0.0,
        max_generated_tokens=64,
    )

    for delta in stream:
        if delta.content:
            print(delta.content, end="", flush=True)
    print()
```

## Responses API

Use `responses()` when you want OpenAI Responses-style output objects, text
deltas, reasoning items, and function-call items.

```python
from orchard.engine.inference_engine import InferenceEngine

MODEL = "google/gemma-4-E2B-it"

with InferenceEngine(load_models=[MODEL]) as engine:
    client = engine.client()
    response = client.responses(
        MODEL,
        input="Explain why local inference is useful in two sentences.",
        temperature=0.0,
        max_output_tokens=96,
    )
    print(response.output_text)
```

For text-only streaming from Responses:

```python
from orchard.engine.inference_engine import InferenceEngine

MODEL = "google/gemma-4-E2B-it"

with InferenceEngine(load_models=[MODEL]) as engine:
    client = engine.client()
    for chunk in client.responses_text(
        MODEL,
        input="Give me three concise debugging tips.",
        temperature=0.0,
        max_output_tokens=96,
    ):
        print(chunk, end="", flush=True)
    print()
```

## Async

Every client path has an async form. Use `achat()`, `aresponses()`, and
`aresponses_text()` inside async services.

```python
import asyncio

from orchard.engine.inference_engine import InferenceEngine

MODEL = "google/gemma-4-E2B-it"


async def main() -> None:
    async with InferenceEngine() as engine:
        await engine.load_model(MODEL)
        client = engine.client()
        response = await client.achat(
            MODEL,
            [{"role": "user", "content": "Say hello from Orchard."}],
            temperature=0.0,
            max_generated_tokens=64,
        )
        print(response.text)


asyncio.run(main())
```

## HTTP Server

Start the server only when another process, `curl`, or an OpenAI-compatible
client needs to talk to Orchard over HTTP. The normal Python path is the client
above.

```bash
orchard serve --model google/gemma-4-E2B-it
```

The default server listens on `http://127.0.0.1:8000`.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="google/gemma-4-E2B-it",
    messages=[{"role": "user", "content": "Hello from Orchard."}],
)
print(response.choices[0].message.content)
```

The server exposes:

| Endpoint | Use |
| --- | --- |
| `POST /v1/chat/completions` | Chat Completions, streaming, batching, tools, structured output |
| `POST /v1/responses` | Responses objects, event streams, reasoning, tool calls, multimodal input |
| `POST /v1/completions` | Text completions |
| `POST /v1/embeddings` | Embeddings for supported models |
| `GET /v1/models` | Loaded model list |
| `GET /health` | Server health |

## Batching

Pass a list of conversations to schedule prompts together. Orchard returns one
response per prompt in order.

```python
from orchard.engine.inference_engine import InferenceEngine

MODEL = "google/gemma-4-E2B-it"

with InferenceEngine(load_models=[MODEL]) as engine:
    client = engine.client()
    responses = client.chat(
        MODEL,
        [
            [{"role": "user", "content": "Say hello politely."}],
            [{"role": "user", "content": "Give me a fun fact about space."}],
        ],
        temperature=0.0,
        max_generated_tokens=24,
    )

    for response in responses:
        print(response.text)
```

Sync, async, streaming, batching, and best-of-N are all supported. See
[`orchard/clients/client.py`](orchard/clients/client.py).

For the HTTP API, send the same shape in `messages`:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-4-E2B-it",
    "messages": [
      [{"role": "user", "content": "Say hello politely."}],
      [{"role": "user", "content": "Give me a fun fact about space."}]
    ],
    "max_completion_tokens": 24,
    "temperature": 0.0
  }'
```

## Multimodal

Use Responses-style content parts for images. Pass images as data URLs.

```python
import base64
from pathlib import Path

from orchard.engine.inference_engine import InferenceEngine

MODEL = "google/gemma-3-4b-it"
IMAGE = Path("apple.jpg")

image_url = "data:image/jpeg;base64," + base64.b64encode(IMAGE.read_bytes()).decode()

with InferenceEngine(load_models=[MODEL]) as engine:
    client = engine.client()
    response = client.responses(
        MODEL,
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What is in this image?"},
                    {"type": "input_image", "image_url": image_url},
                ],
            }
        ],
        temperature=0.0,
        max_output_tokens=96,
    )
    print(response.output_text)
```

## Structured Output

Use JSON Schema when the caller needs machine-readable output.

```python
from orchard.engine.inference_engine import InferenceEngine

MODEL = "google/gemma-4-E2B-it"

schema = {
    "type": "object",
    "properties": {
        "capital": {"type": "string"},
        "population": {"type": "integer"},
    },
    "required": ["capital", "population"],
}

with InferenceEngine(load_models=[MODEL]) as engine:
    client = engine.client()
    response = client.responses(
        MODEL,
        input="What is the capital of France and its approximate population?",
        text={
            "format": {
                "type": "json_schema",
                "name": "city_info",
                "schema": schema,
                "strict": True,
            }
        },
        temperature=0.0,
        max_output_tokens=64,
    )
    print(response.output_text)
```

## Tool Use

Tools use the Responses function schema. Non-streaming responses expose parsed
function calls on `response.tool_calls`.

```python
import json

from orchard.engine.inference_engine import InferenceEngine

MODEL = "google/gemma-4-E2B-it"

weather_tool = {
    "type": "function",
    "name": "get_weather",
    "description": "Get the current weather for a location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name, e.g. San Francisco",
            }
        },
        "required": ["location"],
    },
}

with InferenceEngine(load_models=[MODEL]) as engine:
    client = engine.client()
    response = client.responses(
        MODEL,
        input="What is the weather in San Francisco?",
        tools=[weather_tool],
        tool_choice="required",
        temperature=0.0,
        max_output_tokens=128,
    )

    for call in response.tool_calls:
        print(call.name, json.loads(call.arguments))
```

## Reasoning

Reasoning is model-dependent. For models with native thinking tokens, pass
`reasoning=True` or an effort level.

```python
response = client.responses(
    MODEL,
    input="Solve 23 * 47 and explain the steps briefly.",
    reasoning={"effort": "medium"},
    temperature=0.0,
    max_output_tokens=128,
)
```

Accepted effort values are `minimal`, `low`, `medium`, and `high`.

## Production Use

Orchard is designed for production local services, not just one-off scripts.
The same package covers notebooks, batch jobs, benchmark harnesses, and
long-running agents that keep several models warm.

| Capability | Path |
| --- | --- |
| Multiple loaded models | Start `InferenceEngine(load_models=[...])` or `orchard serve --model model-a model-b` |
| Continuous batching | Send batched prompts or concurrent requests through the same engine process |
| Streaming | Use `stream=True`, `responses_text()`, or Server-Sent Events over HTTP |
| Structured output | Use `response_format` for Chat Completions or `text.format` for Responses |
| Tool use | Use `tools`, `tool_choice`, and `max_tool_calls` |
| Multimodal input | Use Responses content parts with `input_text` and `input_image` |

The engine process is shared by Orchard clients on the machine. Stop it when you
want a clean shutdown:

```bash
orchard engine stop
```

Update the engine binary:

```bash
orchard upgrade stable
```

## Models

Orchard resolves local paths and Hugging Face repos on demand. The currently
tested families include Gemma, Qwen, Llama, and Moondream.

| Mac | Start with |
| --- | --- |
| 8 GB unified memory | `google/gemma-4-E2B-it` |
| 16-32 GB unified memory | `google/gemma-4-E4B-it` or `Qwen/Qwen3.5-4B` |
| 32 GB+ unified memory | `meta-llama/Llama-3.1-8B-Instruct` |

Other model families need a profile in
[Pantheon](https://github.com/TheProxyCompany/Pantheon), which supplies chat
templates, control tokens, and capability metadata.

## Requirements

- macOS 14 or newer
- Apple Silicon Mac
- Python 3.12 or newer
- Disk space for the engine binary and model weights

## Privacy

Inference runs locally on your Mac. Orchard downloads the engine binary and the
model weights you request; prompts and outputs are not sent to a cloud inference
API by Orchard.

## Development

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
pytest
```

For full engine/client verification inside the Proxy Company hyper-repo, run:

```bash
./scripts/pie_cycle.sh --py-only
```

## Related

- [Official Orchard docs](https://docs.theproxycompany.com/orchard/)
- [orchard-rs](https://github.com/TheProxyCompany/orchard-rs) for Rust apps that embed Orchard
- [orchard-swift](https://github.com/TheProxyCompany/orchard-swift) for Swift telemetry
- [Pantheon](https://github.com/TheProxyCompany/Pantheon)
- [Proxy Inference Engine](https://github.com/TheProxyCompany/proxy-inference-engine)

## License

Apache-2.0
