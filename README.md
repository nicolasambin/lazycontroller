# lazycontroller

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/nicolasambin/lazycontroller/blob/main/LICENSE)
![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)
![Status: alpha](https://img.shields.io/badge/status-alpha-orange.svg)

Let a fast decision model answer the structured-output calls your code already makes to an LLM,
by changing one import.

> **Alpha.** lazycontroller works and is tested, but its API may still change between 0.x
> releases.

## Why

Codebases everywhere are full of `OpenAI()` clients asking GPT-class models to categorize a
transaction, flag a message or pick a label, and paying LLM prices and latency for what is really
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

Your existing OpenAI client stays the slow side; the fast side here is Jev, a decision model,
reached through OpenRouter.

```python
import os
from typing import Literal

from pydantic import BaseModel, Field

from lazycontroller import FastConfig, OpenAI

client = OpenAI(  # the slow side, as before: reads OPENAI_API_KEY
    fast=FastConfig(  # the fast side: a System One decision model
        base_url="https://openrouter.ai/api",
        api_key=os.environ["OPENROUTER_API_KEY"],
        model="typesafe/jev-1.14",
    ),
)


class Transaction(BaseModel):
    category: Literal[
        "groceries",
        "dining",
        "transport",
        "shopping",
        "subscriptions",
        "utilities",
        "income",
        "transfer",
    ] = Field(description="What the money was spent on, or where it came from")


for line in ["NETFLIX.COM 866-579-7172 CA", "GITHUB, INC. SAN FRANCISCO"]:
    completion = client.chat.completions.parse(
        model="gpt-6-luna",
        messages=[
            {"role": "system", "content": "Categorize this bank transaction."},
            {"role": "user", "content": line},
        ],
        response_format=Transaction,
    )
    print(line, "->", completion.choices[0].message.parsed.category, "by", completion.model)

# NETFLIX.COM 866-579-7172 CA -> subscriptions by typesafe/jev-1.14-20261014
# GITHUB, INC. SAN FRANCISCO -> ... by gpt-6-luna-...
```

Jev is certain the first line is a subscription (confidence 1.00), so it answers and GPT is
never called. The second could be a subscription or a one-off purchase; Jev is less sure
(around 0.7–0.8, below the default `min_confidence` of 0.9), so the call goes to GPT as it always did.
Either way you get the same `ParsedChatCompletion`.

[`examples/transactions.ipynb`](https://nicolasambin.github.io/lazycontroller/examples/transactions/) goes further: shadow mode on ten
transactions, a sweep over `min_confidence`, `with_fast`, a call that isn't a decision, and the same thing
running locally with Ollama. It needs no Jupyter in your project:

```bash
uv run --with jupyterlab --with matplotlib --env-file .env jupyter lab examples/transactions.ipynb
```

## Choosing a fast model

The fast side is any server speaking the System One API; the slow side is any OpenAI-compatible
API (OpenAI, OpenRouter, Ollama, vLLM, ...).

| Fast model server | `base_url` | `api_key` | Example `model` |
|---|---|---|---|
| TypeSafe | `"https://api.typesafe.ai"` | your TypeSafe key | `"jev-latest"` |
| OpenRouter | `"https://openrouter.ai/api"` | your OpenRouter key | `"typesafe/jev-1.14"` |
| Ollama (local) | `"http://localhost:11434"` | none | `"kev:0.8b-q8_0"` |

Decision models don't all support the same questions: some answer yes/no only.
More in [configuration](https://nicolasambin.github.io/lazycontroller/configuration/#fast-model-servers).

## How it works

For `chat.completions.create` / `parse` and `responses.create` / `parse`:

1. **Is it a decision?** The schema must ask only for closed decisions: `Literal`/`Enum`
   (Choice), `bool` (yes/no), small integer ranges (Score), optionally nested or `| None`. A
   free-text field, tools, streaming, images and the like send the call to the slow model
   untouched.
2. **Ask the fast model.** All fields go in one request, with the conversation as structured
   JSON state and each field's `description` as its question.
3. **Gate on confidence.** If *every* field reaches `min_confidence`, lazycontroller builds a real
   OpenAI response object from the fast answer. Otherwise the original request goes to the slow
   model. It's all or nothing, so you never get a half-fast, half-slow answer.

If the fast model errors or lazycontroller can't read a request, the call goes to the slow model
as if lazycontroller weren't there: it can save you calls, never break them.

Details, and how to write schemas a decision model answers well: [schemas and
requests](https://nicolasambin.github.io/lazycontroller/schemas/).

## Configuration

`OpenAI` / `AsyncOpenAI` take every `openai` argument (the slow side) plus
`fast=FastConfig(...)`:

```python
FastConfig(
    base_url=None,  # the System One server; None turns the fast side off
    api_key=None,  # key for base_url, sent nowhere else
    model=None,  # the decision model; None means the server's default
    min_confidence=0.9,  # every field must reach it, or the call goes to the slow model
    enabled=True,  # False sends everything to the slow model
    shadow=False,  # ask both, return the slow answer, report disagreements
    on_route=None,  # called with a Route after every intercepted call
)
```

To vary settings for part of your code, derive a copy, like `with_options`:

```python
refunds = client.with_fast(min_confidence=0.97)  # a wrong answer costs more here
```

All settings, keys and providers: [configuration](https://nicolasambin.github.io/lazycontroller/configuration/).

## Monitoring and choosing `min_confidence`

Every intercepted call produces a `Route`: who answered (`fast` or `slow`), why, and the
confidence of each field. Collect them to see how much traffic the fast model takes.

The default `min_confidence`, 0.9, is cautious. To choose yours, run with `shadow=True` on real
traffic: both models answer, your users get the slow answer, and each route records whether the
two agreed. A `ShadowReport` turns those routes into coverage and agreement for each `min_confidence`:

```python
from lazycontroller import ShadowReport

report = ShadowReport(routes)
print(report.table())
report.min_confidence(min_agreement=0.98)  # lowest value that keeps 98% agreement
```

Route fields, shadow mode and `ShadowReport`: [routing](https://nicolasambin.github.io/lazycontroller/routing/).

## Data and privacy

- With a hosted fast model, the conversation of every eligible call is also sent to that
  provider (TypeSafe, OpenRouter, ...). Check that this fits your data agreements before
  enabling it. A local fast model keeps everything on your machine.
- In shadow mode, eligible calls go to both providers.
- Free models on some platforms may use requests for training; don't send real data to them.
- Keys go only where they belong: `FastConfig.api_key` is sent only to `FastConfig.base_url`, and no
  key is picked up from the environment for the fast side.

To report a security issue, see [SECURITY.md](https://nicolasambin.github.io/lazycontroller/security/).

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

The documentation site is built with [Zensical](https://zensical.org) from `docs/`:

```bash
uv run --group docs python scripts/notebooks_to_docs.py   # example notebooks -> pages
uv run --group docs zensical serve                        # preview, reloads on changes
```

[`examples/transactions.ipynb`](https://nicolasambin.github.io/lazycontroller/examples/transactions/) is a runnable tour of the library;
`uv run --with jupyterlab --with matplotlib` opens it without adding either to the project.

## License

[Apache License 2.0](https://github.com/nicolasambin/lazycontroller/blob/main/LICENSE).
