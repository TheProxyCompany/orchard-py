# Orchard

Python client for high-performance LLM inference on Apple Silicon.

## Installation

```bash
pip install orchard
```

## Usage

```python
from orchard import Client

client = Client()

response = client.chat(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.text)
```

### Streaming

```python
for delta in client.chat(model="...", messages=[...], stream=True):
    print(delta.content, end="", flush=True)
```

### Batch Inference

```python
responses = client.chat_batch(
    model="...",
    conversations=[
        [{"role": "user", "content": "Question 1"}],
        [{"role": "user", "content": "Question 2"}],
    ],
)
```

## Model Profiles

Chat templates and control tokens are loaded from the [Pantheon](https://github.com/TheProxyCompany/Pantheon) submodule at `orchard/formatter/profiles/`. This provides a single source of truth shared across all Orchard SDKs (Python, Rust, Swift). See that repo for the list of supported model families.

## Requirements

- Python 3.10+
- macOS 14+ (Apple Silicon)
- PIE (Proxy Inference Engine)

## Related

- [orchard-rs](https://github.com/TheProxyCompany/orchard-rs) - Rust client
- [orchard-swift](https://github.com/TheProxyCompany/orchard-swift) - Swift client
- [Pantheon](https://github.com/TheProxyCompany/Pantheon) - Model profiles

## License

Apache-2.0
