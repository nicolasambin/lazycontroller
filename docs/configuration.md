# Configuration

## The two sides

lazycontroller's `OpenAI` and `AsyncOpenAI` take every argument of `openai.OpenAI` /
`openai.AsyncOpenAI`, which configure the **slow** side, plus `lazy=LazyConfig(...)` for the
**fast** side and the routing.

```python
from lazycontroller import LazyConfig, OpenAI

client = OpenAI(
    api_key="...",  # slow side: any openai.OpenAI argument (base_url, timeout, ...)
    lazy=LazyConfig(  # fast side and routing
        fast_model="kev:0.8b-q8_0",
        fast_base_url="http://localhost:11434",
        threshold=0.8,
    ),
)
```

The model in each call (`model="gpt-6-luna"`) is the slow model: the one your code already uses.

## `LazyConfig`

| Setting | Default | Meaning |
|---|---|---|
| `enabled` | `True` | `False` sends every call straight to the slow model. |
| `threshold` | `0.9` | Every field's confidence must reach this for the fast model's answer to be used. See [routing](routing.md#choosing-a-threshold). |
| `fast_base_url` | `None` | The System One server. Without it (or `fast_client`) the fast side is off. |
| `fast_api_key` | `None` | Key for `fast_base_url`. See [keys](#keys) below. |
| `fast_model` | `None` | The decision model. `None` uses the server's default. |
| `fast_client` | `None` | A ready `typesafe_sdk.TypeSafeClient` (or `AsyncTypeSafeClient`) to use instead of creating one. |
| `shadow` | `False` | Ask both models, always return the slow answer, report disagreements. See [shadow mode](routing.md#shadow-mode). |
| `on_route` | `None` | Called with a [`Route`](routing.md#route) after every intercepted call. |

`LazyConfig` is immutable; an unknown setting raises `TypeError` and a threshold outside 0–1
raises `ValueError`.

## Fast model servers

The fast model is any server speaking the System One API:

| Server | `fast_base_url` | `fast_api_key` | Example `fast_model` |
|---|---|---|---|
| TypeSafe | `"https://api.typesafe.ai"` | your TypeSafe key | `"jev-latest"` |
| OpenRouter | `"https://openrouter.ai/api"` | your OpenRouter key | `"typesafe/jev-1.14"` |
| Ollama (local) | `"http://localhost:11434"` | none | `"kev:0.8b-q8_0"` |

Every server is configured the same way: there is no default, and nothing is read from the
environment, so without `fast_base_url` every call goes to the slow model
(`reason="disabled"`).

Decision models don't all support the same question types; check yours (see
[schemas](schemas.md#writing-schemas-a-decision-model-answers-well)). The state is sent as
structured JSON, so servers that accept only a string state can't serve as the fast model.

## Keys

- **Fast side:** `fast_api_key` is sent to `fast_base_url` and nowhere else. Environment
  variables such as `TYPESAFE_API_KEY` are not read. Local servers need no key.
- **Slow side:** as in `openai.OpenAI` (`api_key` or `OPENAI_API_KEY`), except that a local
  `base_url` (`localhost`, `127.0.0.1`, `::1`, `*.localhost`) needs no key. A remote server
  without a key raises the SDK's usual error.

## Varying settings with `with_lazy`

Settings live on the client, so call sites stay as they are. To change them for part of your
code, derive a copy, as `with_options` does for the SDK's own options:

```python
refunds = client.with_lazy(threshold=0.97)  # a wrong answer costs more here
tagging = client.with_lazy(threshold=0.6)
plain = client.with_lazy(enabled=False)  # this code path always uses the slow model
```

Copies share the connection to the fast model when its server and key are unchanged, and the
original client keeps its settings. `with_options` on a copy keeps the copy's routing settings.

## What stays the stock SDK

Only `chat.completions.create` / `parse` and `responses.create` / `parse` are intercepted.
Everything else (`embeddings`, `files`, `models`, `with_raw_response`, streaming, ...) is
`openai` itself, and an `isinstance(client, openai.OpenAI)` check still passes.
