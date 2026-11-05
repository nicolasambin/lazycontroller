# lazycontroller

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)
![Status: alpha](https://img.shields.io/badge/status-alpha-orange.svg)

Let a fast decision model answer the structured-output calls your code already makes to an LLM,
by changing one import.

> **Alpha.** lazycontroller works and is tested, but its API may still change between 0.x
> releases.

## Why

Codebases everywhere are full of `OpenAI()` clients asking GPT-class models to classify a
ticket, flag a message or pick a category, and paying LLM prices and latency for what is really
a decision. Decision models now do that job in milliseconds, for a fraction of the cost, and
tell you how sure they are. You don't need to rewrite anything to use them: change one import.

```python
# from openai import OpenAI
from lazycontroller import OpenAI
```

When the decision model is confident, it answers. When it isn't, or when the call isn't a
decision at all, the request goes to the model you were already calling, exactly as before.

## Why "lazycontroller"

In *Thinking, Fast and Slow*, Daniel Kahneman calls the deliberate mind, his System 2, a *lazy
controller*: it is in charge, but it lets the fast, intuitive System 1 handle whatever it can,
and steps in only when that isn't enough. lazycontroller works the same way. A fast decision
model (a "System One" model) answers the calls it is sure about; the slow LLM you were already
using is called only when it isn't.

## Install

```bash
pip install lazycontroller
```

or `uv add lazycontroller`. Requires Python 3.13+.

## Quickstart

This runs entirely on your machine with [Ollama](https://ollama.com): `kev`, a decision model,
as the fast side and `qwen3.5:4b` as the slow one.

```bash
ollama pull kev:0.8b-q8_0
ollama pull qwen3.5:4b
```

```python
from typing import Literal

from pydantic import BaseModel, Field

from lazycontroller import LazyConfig, OpenAI, Route

routes: list[Route] = []
client = OpenAI(
    base_url="http://localhost:11434/v1",  # the slow model: any OpenAI-compatible API
    lazy=LazyConfig(
        fast_model="kev:0.8b-q8_0",  # the fast model: a System One decision model
        fast_base_url="http://localhost:11434",
        threshold=0.7,
        on_route=routes.append,
    ),
)


class Ticket(BaseModel):
    team: Literal["billing", "technical", "sales"] = Field(
        description="Which team should handle it"
    )


completion = client.chat.completions.parse(
    model="qwen3.5:4b",
    messages=[
        {"role": "system", "content": "Triage this customer support ticket."},
        {"role": "user", "content": "Every API call has returned 500 for the last 20 minutes!"},
    ],
    response_format=Ticket,
)

print(completion.choices[0].message.parsed)  # team='technical'
print(completion.model, routes[-1].reason)  # kev:0.8b-q8_0 confident
```

With OpenAI as the slow side, drop `base_url`, keep your `model="gpt-5"`, and point
`LazyConfig` at the decision model you want (below). The rest of your code doesn't change:
whoever answers, you get the same `ParsedChatCompletion`.

## Choosing a fast model

The fast side is any server speaking the System One API; the slow side is any OpenAI-compatible
API (OpenAI, OpenRouter, Ollama, vLLM, ...).

| Fast model server | `fast_base_url` | `fast_api_key` | Example `fast_model` |
|---|---|---|---|
| TypeSafe | (default) | `TYPESAFE_API_KEY` | `None` (`jev-latest`) |
| OpenRouter | `"https://openrouter.ai/api"` | your OpenRouter key | `"typesafe/jev-1.13"` |
| Ollama (local) | `"http://localhost:11434"` | none | `"kev:0.8b-q8_0"` |

Decision models don't all support the same questions: some answer yes/no only.
More in [configuration](docs/configuration.md#fast-model-servers).

## How it works

For `chat.completions.create` / `parse` and `responses.create` / `parse`:

1. **Is it a decision?** The schema must ask only for closed decisions: `Literal`/`Enum`
   (Choice), `bool` (yes/no), small integer ranges (Score), optionally nested or `| None`. A
   free-text field, tools, streaming, images and the like send the call to the slow model
   untouched.
2. **Ask the fast model.** All fields go in one request, with the conversation as a plain-text
   transcript and each field's `description` as its question.
3. **Gate on confidence.** If *every* field reaches the threshold, lazycontroller builds a real
   OpenAI response object from the fast answer. Otherwise the original request goes to the slow
   model. It's all or nothing, so you never get a half-fast, half-slow answer.

If the fast model errors or lazycontroller can't read a request, the call goes to the slow model
as if lazycontroller weren't there: it can save you calls, never break them.

Details, and how to write schemas a decision model answers well: [schemas and
requests](docs/schemas.md).

## Configuration

`OpenAI` / `AsyncOpenAI` take every `openai` argument (the slow side) plus
`lazy=LazyConfig(...)`:

```python
LazyConfig(
    enabled=True,  # False sends everything to the slow model
    threshold=0.9,  # min confidence of every field
    fast_model=None,  # the decision model; None means the server's default
    fast_base_url=None,  # None means TypeSafe
    fast_api_key=None,  # defaults to TYPESAFE_API_KEY on TypeSafe
    shadow=False,  # ask both, return the slow answer, report disagreements
    on_route=None,  # called with a Route after every intercepted call
)
```

To vary settings for part of your code, derive a copy, like `with_options`:

```python
refunds = client.with_lazy(threshold=0.97)  # a wrong answer costs more here
```

All settings, keys and providers: [configuration](docs/configuration.md).

## Monitoring and thresholds

Every intercepted call produces a `Route`: who answered (`fast` or `slow`), why, and the
confidence of each field. Collect them to see how much traffic the fast model takes.

The default threshold, 0.9, is cautious. To choose yours, run with `shadow=True` on real
traffic: both models answer, your users get the slow answer, and each route lists where the two
disagreed. Then pick the lowest threshold whose disagreement rate you can accept.

Route fields, shadow mode and a threshold sweep: [routing](docs/routing.md).

## Data and privacy

- With a hosted fast model, the conversation of every eligible call is also sent to that
  provider (TypeSafe, OpenRouter, ...). Check that this fits your data agreements before
  enabling it. A local fast model keeps everything on your machine.
- In shadow mode, eligible calls go to both providers.
- Free models on some platforms may use requests for training; don't send real data to them.
- Keys go only where they belong: `TYPESAFE_API_KEY` is sent only to TypeSafe, and a custom
  `fast_base_url` receives only the `fast_api_key` you give it.

## Limitations

- Forced tool calls (`tool_choice` with a function) aren't treated as structured output yet,
  so they always go to the slow model.
- Only OpenAI-style clients for now: no `from lazycontroller import Anthropic` yet.
- `with_raw_response`, `with_streaming_response` and streaming always go to the slow model.
- Answers from the fast model aren't stored by the slow model's provider (`store=True` has no
  effect on them).

## Development

The project is managed with [uv](https://docs.astral.sh/uv/). CI runs lint and tests on Python
3.13–3.15 for every pull request.

```bash
uv run pytest                              # tests (both models are faked, no keys needed)
uv run --python 3.13 --isolated pytest     # tests on the minimum version
uv run ruff check && uv run ruff format    # lint and format
```

[`examples/ollama_local.py`](examples/ollama_local.py) compares a local decision model with a
local LLM in shadow mode.

## License

[Apache License 2.0](LICENSE).
