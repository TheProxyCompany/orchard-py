# Orchard

**100% local, OpenAI-compatible LLM inference for Apple Silicon.** Multi-model serving, prefix caching, continuous batching. No cloud APIs, no data leaves your machine.

`macOS 14+` · `Apple Silicon (M1+)` · `Python 3.12+` · Apache-2.0

## Features

- **Drop-in OpenAI API** — `/v1/chat/completions`, `/v1/responses`, `/v1/embeddings`, `/v1/models`
- **Fast** — C++ inference engine with prefix caching and continuous batching
- **Multi-model** — load Qwen, Llama, and Gemma side-by-side; swap between them per request
- **Multimodal** — vision, tool calling, thinking; native where the model was trained, grammar-constrained where it wasn't
- **OpenAI Responses API** — streaming events for reasoning, tool calls, and messages
- **Use from anything** — curl, Python (openai SDK), Rust ([orchard-rs](https://github.com/TheProxyCompany/orchard-rs)), or any OpenAI-compatible client

## Getting started

```bash
pip install orchard
orchard serve --model google/gemma-4-E4B-it
```

First run downloads the PIE engine binary (~2 GB) and the model weights from HuggingFace. Subsequent runs start in seconds.

Then point anything at `http://localhost:8000/v1`:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-4-E4B-it",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Or with the OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="google/gemma-4-E4B-it",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

## Supported models

Orchard downloads open-weights models from HuggingFace on demand. Any model whose family has a profile in [Pantheon](https://github.com/TheProxyCompany/Pantheon) works out of the box.

| Model | Size (BF16) | Modalities | Best for |
|---|---|---|---|
| **google/gemma-4-E4B-it** (default) | ~8 GB | text, vision | Multimodal, native thinking, tool calls |
| Qwen/Qwen3.5-4B | ~9 GB | text | 256k context, native thinking, tool calls |
| meta-llama/Llama-3.1-8B-Instruct | ~16 GB | text | General-purpose, trained tool calls |
| google/gemma-3-4b-it | ~8 GB | text, vision | Multimodal chat |
| google/gemma-4-E2B-it | ~5 GB | text, vision | Fits on 8 GB Macs |
| moondream/moondream3-preview | ~9 GB | vision | Pointing, detection, captioning |

**By hardware**

| Your Mac | Recommended |
|---|---|
| M1 / M2 / M3 (8 GB) | `google/gemma-4-E2B-it` |
| M-Pro / Max (16–32 GB) | `google/gemma-4-E4B-it`, `Qwen/Qwen3.5-4B` |
| M-Max / Ultra (32+ GB) | `meta-llama/Llama-3.1-8B-Instruct` + a small model hot-loaded |

> Need quantized weights? Pass any `mlx-community/...` repo directly. Orchard resolves any HuggingFace repo whose architecture belongs to a supported family.

## OpenAI Responses API

Orchard implements the [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses) — the successor to Chat Completions. You get structured streaming events for reasoning, tool calls, and messages, plus per-item state and lifecycle events.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
stream = client.responses.create(
    model="google/gemma-4-E4B-it",
    input="Explain quantum tunneling in three sentences.",
    reasoning={"effort": "medium"},
    stream=True,
)
for event in stream:
    print(event)
```

Supports `response.output_text.delta`, `response.reasoning.delta`, `response.function_call_arguments.delta`, structured output via JSON Schema, and `max_tool_calls` for bounded tool loops.

## Python SDK (no HTTP)

For embedded use — skip the HTTP layer:

```python
from orchard.engine.inference_engine import InferenceEngine

async with InferenceEngine() as engine:
    client = engine.client()
    response = await client.achat(
        "google/gemma-4-E4B-it",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    print(response.text)
```

Sync, async, streaming, batching, and best-of-N are all supported. See [`orchard/clients/client.py`](orchard/clients/client.py).

## Privacy

Orchard runs entirely on your Mac. No telemetry, no analytics, no phone-home.

| Surface | Status | What it does |
|---|---|---|
| Inference | ✅ Local | All generation on-device via PIE (C++, Metal) |
| Chat templates | ✅ Local | Rendered from Pantheon profiles bundled in the package |
| Model weights | ✅ One-time | HuggingFace Hub → `~/.cache/huggingface/` |
| Engine binary | ✅ One-time | GitHub release → `~/.orchard/bin/` |
| Telemetry | ✅ None | No tracking SDKs — verify with `grep -r analytics orchard/` |

## How it works

Orchard is the Python layer over a stack built for Apple Silicon:

- [**PIE**](https://github.com/TheProxyCompany/proxy-inference-engine) — C++ inference engine: prefix caching, continuous batching, multi-model scheduling
- [**PAL**](https://github.com/TheProxyCompany/proxy-attention-lab) — Metal GPU kernels
- [**PSE**](https://github.com/TheProxyCompany/proxy-state-engine) — grammar-constrained generation for tool calls, structured output, and thinking
- [**Pantheon**](https://github.com/TheProxyCompany/Pantheon) — chat templates and capability manifests, shared across all Orchard SDKs

The Python package handles IPC, model resolution, HuggingFace downloads, prompt rendering, and the FastAPI server.

## CLI

```bash
orchard serve --model <hf-repo> [--host 127.0.0.1] [--port 8000]
orchard serve --models model-a model-b model-c    # preload multiple
orchard upgrade [stable|nightly]                  # update engine binary
orchard engine stop                               # kill background engine
```

## Requirements

- macOS 14+, Apple Silicon (M1 or newer)
- Python 3.12+
- ~2 GB free disk for the engine binary, plus model weights

## Related

- [orchard-rs](https://github.com/TheProxyCompany/orchard-rs) — Rust client
- [orchard-swift](https://github.com/TheProxyCompany/orchard-swift) — Swift telemetry client
- [Pantheon](https://github.com/TheProxyCompany/Pantheon) — model profiles
- [PIE](https://github.com/TheProxyCompany/proxy-inference-engine) — the engine

## License

Apache-2.0
