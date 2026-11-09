# Schemas and requests

How lazycontroller decides whether the fast model can take a request, what it asks it, and how
to write schemas that a decision model answers well.

## From schema to questions

A request is a candidate only if its structured-output schema (`response_format`,
`text_format` or `text.format`) asks for closed decisions and nothing else. Each leaf field
becomes one question, and all questions go to the fast model in a single request.

| Schema field | Question | Value returned | Confidence |
|---|---|---|---|
| `Literal[...]`, `Enum`, `{"enum": [...]}` | Choice, one option per value | the chosen value | the model's `confidence` |
| `bool` | Noul (yes/no) | `p >= 0.5` | `\|2p − 1\|` |
| `int` with `ge`/`le` (or `minimum`/`maximum`), 2–11 values | Score, one level per value | the most likely level | the model's `confidence` |
| `X \| None` where `X` is an enum or `bool` | Choice with an extra `null` option ("Not stated or not applicable") | the chosen value or `None` | the model's `confidence` |
| nested model / object | one question per leaf, path `parent.child` | rebuilt as nested JSON | — |
| `Literal["x"]` / `const` | none | filled in | — |

A few details:

- **Instructions.** Each question reads *"Decide the `field` field of the `<SchemaName>` the
  conversation asks for: `<description>`"*. Without a `description`, the field name is used
  (`is_urgent` becomes "is urgent").
- **Score levels** are the integers themselves (`severity = 1`, `severity = 2`, ...). The
  answer is the most likely level, not the expected value, because the schema wants one of its
  integers.
- **Noul confidence.** Yes/no answers carry a probability `p`, not a confidence; `|2p − 1|` puts
  them on the same 0–1 scale as Choice (0 at `p = 0.5`, 1 at `p = 0` or `1`).

## What sends a request to the slow model

Anything a decision model can't answer, or any option it can't honour, sends the request to the
slow model untouched. `Route.reason` is `ineligible` and `Route.detail` says why.

**In the schema:**

- free-text `str`, arrays, `float`, integers without both bounds or with more than 11 values
- unions of several types (other than `X | None`)
- any JSON Schema keyword whose meaning isn't honoured, such as `multipleOf`, `pattern`,
  `format`, `minLength`, `patternProperties` or an open `additionalProperties` schema.
  Annotations (`title`, `description`, `default`, `examples`, ...) are fine.

**In the request:**

- a message that isn't plain text (images, audio, files), a `tool` message or a tool call in
  the history, a non-message item in `responses` input
- options other than sampling, limits and bookkeeping: `tools`, `stream=True`, `logprobs`,
  `n > 1`, `audio`, `prediction`, `previous_response_id`, `background`, `extra_headers`,
  `extra_query`, `extra_body`, and so on. The parameters that don't change what a correct answer
  is are ignored: `model`, `temperature`, `top_p`, `seed`, `max_tokens`,
  `max_completion_tokens`, `max_output_tokens`, `frequency_penalty`, `presence_penalty`,
  `reasoning`, `reasoning_effort`, `verbosity`, `service_tier`, `metadata`, `store`, `user`,
  `safety_identifier`, `prompt_cache_key`, `prompt_cache_retention`, `truncation`, `timeout`.
- `with_raw_response` and `with_streaming_response` calls

## The state: the conversation as JSON

The conversation goes to the fast model as a structured state, one entry per message:

```json
{
  "conversation": [
    {"role": "system", "content": "Categorize this bank transaction."},
    {"role": "user", "content": "NETFLIX.COM 866-579-7172 CA"}
  ]
}
```

`developer` messages appear as `system`, and multi-part text content is joined line by line.
Structured state is what System One servers recommend; a server that accepts only a string
state can't serve as the fast model (its calls fail and fall back to the slow model).

## Writing schemas a decision model answers well

- **Describe every field.** The `description` is the question the decision model answers.
  "What the money was spent on" beats a bare `category`.
- **Prefer labels to numbers.** `Literal["calm", "annoyed", "angry"]` gives the model three
  meaningful options; `int = Field(ge=1, le=3)` gives it `frustration = 1/2/3`. In our tests,
  descriptive labels produced more sensible answers.
- **Keep generative fields apart.** One free-text field (a `summary`, a `reason`) makes the whole
  schema ineligible. If you need both a decision and a text, consider two calls: the decision
  can then go to the fast model.
- **Fewer fields, higher odds.** Every field has to clear the threshold, so a schema with ten
  fields is answered by the fast model less often than one with two. Yes/no fields are the usual
  bottleneck: `|2p − 1|` turns a probability of 0.8 into a confidence of 0.6, so a single `bool`
  can hold back a request whose other fields the model is sure of.
- **Check what your fast model supports.** Some decision models answer only some question
  types, for instance only yes/no. A request with an unsupported type fails on the fast side
  and falls back to the slow model (`reason="fast_model_error"`).
