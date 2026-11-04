# lazycontroller

Most structured-output calls to an LLM are really decisions: pick a category, flag yes or no,
rate 1–5. Decision models such as [Jev](https://docs.typesafe.ai/) or a local Kev answer those
faster and far cheaper, and say how sure they are. lazycontroller is a drop-in `openai` client
that lets a **fast** decision model answer the calls it is confident about and sends everything
else to the **slow** model you were already calling.

```python
# from openai import OpenAI
from lazycontroller import OpenAI

client = OpenAI()  # reads OPENAI_API_KEY, and TYPESAFE_API_KEY for Jev


class Ticket(BaseModel):
    team: Literal["billing", "technical", "sales"]
    is_urgent: bool
    severity: int = Field(ge=1, le=5)


completion = client.chat.completions.parse(
    model="gpt-5",  # the slow model
    messages=[
        {"role": "system", "content": "Triage the support ticket."},
        {"role": "user", "content": ticket_text},
    ],
    response_format=Ticket,
)
ticket = completion.choices[0].message.parsed  # same object, whoever answered
```

`completion.model` is the fast model (`"jev-…"`) or the slow one (`"gpt-5…"`); nothing else
changes for the calling code.

Either side can be hosted or local. The fast model is any server speaking the System One API
(TypeSafe, Ollama, llama.cpp, another provider); the slow model is any OpenAI-compatible API
(OpenAI, Ollama, vLLM, OpenRouter, ...).

## How it works

For `chat.completions.create` / `parse` and `responses.create` / `parse`:

1. **Is it a decision?** The request needs a JSON schema (`response_format`, `text_format`
   or `text.format`) whose fields are all closed decisions:

   | Schema field | Decision question | Value |
   |---|---|---|
   | `Literal`, `Enum`, `{"enum": [...]}` | Choice | the chosen option |
   | `bool` | Noul | `p >= 0.5` |
   | `int` with `ge`/`le`, at most 11 values | Score, one level per value | the most likely level |
   | `X \| None` of an enum or bool | Choice with a `null` option | |
   | nested models | one question per leaf field | |
   | `Literal["x"]` / `const` | none | filled in |

   A free-text `str`, a list, a `float`, an open-ended `int`, a union, a constraint such as
   `multiple_of` or `pattern`, images in the messages, or options such as `tools`, `stream`,
   `logprobs`, `n > 1`, `extra_body` or `previous_response_id` mean the request needs a
   generative model, so it goes to the slow model untouched.
2. **Ask the fast model.** All fields go in one request, with the conversation as state and
   each field's name and `description` as its instructions. Good field descriptions help a
   decision model the same way they help an LLM.
3. **Gate on confidence.** If *every* field's confidence reaches the threshold (default 0.9),
   lazycontroller builds a real `ChatCompletion` / `ParsedChatCompletion` / `Response` /
   `ParsedResponse` from the fast model's answer. Otherwise the original request goes to the
   slow model. It is all or nothing: a half-fast, half-slow object could be inconsistent, and
   the fast call costs a tiny fraction of the slow one.

If the fast model errors, isn't configured, or lazycontroller can't read a request, the call
goes to the slow model as if lazycontroller weren't there.

## Configuration

```python
from lazycontroller import LazyConfig, OpenAI

client = OpenAI(
    api_key="...",  # any openai.OpenAI argument: this is the slow side
    lazy=LazyConfig(
        enabled=True,  # False sends everything to the slow model
        threshold=0.9,  # min confidence of every field
        fast_model=None,  # None means the server's default (jev-latest on TypeSafe)
        fast_base_url=None,  # None means TypeSafe; see "Local and other providers"
        fast_api_key=None,  # defaults to TYPESAFE_API_KEY on TypeSafe
        shadow=False,  # see below
        on_route=print,  # called with a Route for every intercepted call
    ),
)
```

`AsyncOpenAI` takes the same arguments.

The model you pass to each call is the one you'd use without lazycontroller, so call sites stay
as they are. Routing settings live on the client; to vary them for part of your code, derive a
copy with `with_lazy`, the routing counterpart of `with_options`:

```python
refunds = client.with_lazy(threshold=0.97)  # a wrong answer costs more here
tagging = client.with_lazy(threshold=0.6)
plain = client.with_lazy(enabled=False)  # this code path always uses the slow model
```

Copies share the connection to the fast model, and the original client keeps its settings.

Each `Route` says who answered (`answered_by`: `fast` or `slow`), why (`reason`: `confident`,
`low_confidence`, `ineligible`, `fast_model_error`, `disabled` or `shadow`), both models, the
confidence of every field, the fast model's token usage, and for ineligible requests the field
or option that ruled the fast model out (`detail`). Log it to see how much traffic the fast
model takes.

### Shadow mode

`LazyConfig(shadow=True)` asks both models, always returns the slow model's answer, and reports
in `Route.shadow_mismatches` the fields where they disagreed, next to the fast model's
confidence. Run it on real traffic to pick a threshold before letting the fast model answer.

### Local and other providers

Ollama serves decision models such as `kev` on the System One API, and any model on its
OpenAI-compatible API, so both sides can run locally. A local server needs no API key:

```python
client = OpenAI(
    base_url="http://localhost:11434/v1",
    lazy=LazyConfig(fast_model="kev:0.8b-q8_0", fast_base_url="http://localhost:11434"),
)
```

For a decision model hosted by someone other than TypeSafe, set `fast_base_url` and its
`fast_api_key`. `TYPESAFE_API_KEY` is only ever sent to TypeSafe.

[`examples/ollama_local.py`](examples/ollama_local.py) runs a few tickets locally in shadow mode.

## What it doesn't do (yet)

- Forced tool calls (`tool_choice` with a function) as a way of getting structured output.
- `from lazycontroller import Anthropic` for Claude.
- `with_raw_response`, `with_streaming_response` and streaming always go to the slow model.
- Answers from the fast model aren't stored by the slow model's provider (`store=True` has no
  effect on them).

## Development

Requires Python 3.13+. The project is managed with [uv](https://docs.astral.sh/uv/).

```bash
uv run pytest                              # tests (both models are faked, no keys needed)
uv run --python 3.13 --isolated pytest     # tests on the minimum version
uv run ruff check && uv run ruff format    # lint and format
```
