# Configuration

## The two sides

lazycontroller's `OpenAI` and `AsyncOpenAI` take every argument of `openai.OpenAI` /
`openai.AsyncOpenAI`, which configure the **slow** side, plus `fast=FastConfig(...)` for the
**fast** side: which decision model to ask, and when to trust its answer.

```python
from lazycontroller import FastConfig, OpenAI

client = OpenAI(
    api_key="...",  # slow side: any openai.OpenAI argument (base_url, timeout, ...)
    fast=FastConfig(  # fast side: same names as openai.OpenAI, for the decision model
        base_url="http://localhost:11434",
        model="kev:0.8b-q8_0",
        min_confidence=0.8,
    ),
)
```

The model in each call (`model="gpt-6-luna"`) is the slow model: the one your code already uses.

## `FastConfig`

| Setting | Default | Meaning |
|---|---|---|
| `base_url` | `None` | The System One server. Without it (or `client`) the fast side is off. |
| `api_key` | `None` | Key for `base_url`. See [keys](#keys) below. |
| `model` | `None` | The decision model. `None` uses the server's default. |
| `min_confidence` | `0.9` | The fast model's answer is used only if every field's confidence reaches this; otherwise the call goes on to the slow model. See [choosing it](routing.md#choosing-the-minimum-confidence). |
| `enabled` | `True` | `False` sends every call straight to the slow model. |
| `shadow` | `False` | Ask both models, always return the slow answer, report disagreements. See [shadow mode](routing.md#shadow-mode). |
| `on_route` | `None` | Called with a [`Route`](routing.md#route) after every intercepted call. |
| `client` | `None` | A ready `typesafe_sdk.TypeSafeClient` (or `AsyncTypeSafeClient`) to use instead of creating one. |

`FastConfig` is immutable; an unknown setting raises `TypeError` and a `min_confidence`
outside 0–1 raises `ValueError`.

## Fast model servers

The fast model is any server speaking the System One API:

| Server | `base_url` | `api_key` | Example `model` |
|---|---|---|---|
| TypeSafe | `"https://api.typesafe.ai"` | your TypeSafe key | `"jev-latest"` |
| OpenRouter | `"https://openrouter.ai/api"` | your OpenRouter key | `"typesafe/jev-1.14"` |
| Ollama (local) | `"http://localhost:11434"` | none | `"kev:0.8b-q8_0"` |

Every server is configured the same way: there is no default, and nothing is read from the
environment, so without `base_url` every call goes to the slow model
(`reason="disabled"`).

Decision models don't all support the same question types; check yours (see
[schemas](schemas.md#writing-schemas-a-decision-model-answers-well)). The state is sent as
structured JSON, so servers that accept only a string state can't serve as the fast model.

## Keys

- **Fast side:** `FastConfig.api_key` is sent to `FastConfig.base_url` and nowhere else.
  Environment variables such as `TYPESAFE_API_KEY` are not read. Local servers need no key.
- **Slow side:** as in `openai.OpenAI` (`api_key` or `OPENAI_API_KEY`), except that a local
  `base_url` (`localhost`, `127.0.0.1`, `::1`, `*.localhost`) needs no key. A remote server
  without a key raises the SDK's usual error.

## Varying settings with `with_fast`

Settings live on the client, so call sites stay as they are. To change them for part of your
code, derive a copy, as `with_options` does for the SDK's own options:

```python
refunds = client.with_fast(min_confidence=0.97)  # a wrong answer costs more here
tagging = client.with_fast(min_confidence=0.6)
plain = client.with_fast(enabled=False)  # this code path always uses the slow model
```

Copies share the connection to the fast model when its server and key are unchanged, and the
original client keeps its settings. `with_options` on a copy keeps the copy's `FastConfig`.

## What stays the stock SDK

Only `chat.completions.create` / `parse` and `responses.create` / `parse` are intercepted.
Everything else (`embeddings`, `files`, `models`, `with_raw_response`, streaming, ...) is
`openai` itself, and an `isinstance(client, openai.OpenAI)` check still passes.
