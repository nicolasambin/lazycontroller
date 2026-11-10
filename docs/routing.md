# Routing and monitoring

## Route

After every intercepted call, `FastConfig.on_route` receives a `Route`:

| Field | Meaning |
|---|---|
| `api` | `"chat.completions"` or `"responses"` |
| `answered_by` | `"fast"` or `"slow"` |
| `reason` | why; see below |
| `slow_model` | the model the call asked for |
| `fast_model` | the decision model that answered, when it was asked |
| `confidence` | the lowest field confidence, compared with `min_confidence` |
| `field_confidences` | confidence per field, keyed by dotted path |
| `fast_output` | the fast model's answer as a JSON object, whenever it was asked, even if the slow model answered in the end |
| `fast_input_tokens` | tokens the fast model billed |
| `detail` | why the request was ineligible, or the fast model's error |
| `shadow_mismatches` | in shadow mode, the fields where the two models disagreed |

| `reason` | `answered_by` | When |
|---|---|---|
| `confident` | fast | every field reached `min_confidence` |
| `low_confidence` | slow | at least one field didn't |
| `ineligible` | slow | the schema or the request needs the slow model ([details](schemas.md#what-sends-a-request-to-the-slow-model)) |
| `fast_model_error` | slow | the fast model failed, or its answer was rejected (for example by a validator in your Pydantic model) |
| `disabled` | slow | `enabled=False`, or no `base_url` for the fast model |
| `shadow` | slow | shadow mode |

## Watching the traffic

`on_route` is any callable; a list's `append` collects routes for inspection.

```python
from collections import Counter
from lazycontroller import FastConfig, OpenAI, Route

seen = Counter()


def track(route: Route) -> None:
    seen[route.answered_by, route.reason] += 1
    if route.reason == "fast_model_error":
        logger.warning("fast model failed: %s", route.detail)


client = OpenAI(fast=FastConfig(base_url=..., on_route=track))
...
share = sum(n for (by, _), n in seen.items() if by == "fast") / sum(seen.values())
```

`detail` on `ineligible` routes tells you which field or option keeps a call away from the
fast model; it's often a single free-text field or `tools`.

## What a fast answer looks like

When the fast model answers, you get the same object types as from the SDK
(`ChatCompletion`, `ParsedChatCompletion`, `Response`, `ParsedResponse`), with:

- `model`: the decision model (e.g. `"kev:0.8b-q8_0"`), so it's visible who answered
- `id`: prefixed `chatcmpl-lazy-` / `resp_lazy_`
- `usage`: the fast model's tokens
- `content` / `output_text`: the JSON object; `.parsed` / `output_parsed`: your model instance

The answer isn't stored by the slow model's provider, so `store=True` has no effect on it.

## Shadow mode

`FastConfig(shadow=True)` asks both models, always returns the slow model's answer, and fills
`Route.shadow_mismatches` with the fields where the two disagreed; `Route.fast_output` holds
what the fast model would have answered. Nothing changes for your
users, and you get the data to decide whether, and from which confidence, to let the fast model
answer.

## Choosing the minimum confidence

`min_confidence` is the confidence every field of the fast model's answer must reach for that
answer to be used. The default, 0.9, is cautious. The right value depends on the decision model and on what a
wrong answer costs you.

1. Run in shadow mode on real traffic for a while, collecting routes.
2. Feed them to a `ShadowReport`: for any `min_confidence` it tells you how much traffic the fast
   model would take and how often it would agree with the slow one.
3. Pick the value, turn shadow off, and use `with_fast` to keep a higher one where
   mistakes are expensive.

```python
from lazycontroller import ShadowReport

report = ShadowReport(routes)  # non-shadow routes are ignored
print(report.table())

report.suggest_min_confidence(min_agreement=0.98)  # e.g. 0.78, or None if there isn't enough data
report.at(0.78)  # Point(min_confidence=0.78, taken=14, agreed=14, total=22)
```

```
22 shadow calls

confidence band   calls  agreement
  0.90-1.00           9  9/9 (100%)
  0.70-0.90           6  5/6 (83%)
  0.00-0.70           7  3/7 (43%)

min confidence   fast takes      agreement
  0.50           19/22 (86%)     16/19 (84%)
  0.60           17/22 (77%)     15/17 (88%)
  0.70           15/22 (68%)     14/15 (93%)
  0.80           12/22 (55%)     12/12 (100%)
  0.90           9/22 (41%)      9/9 (100%)
```

| Method | Returns |
|---|---|
| `table(values=...)` | the plain-text summary above |
| `suggest_min_confidence(min_agreement, min_samples=20)` | the lowest `min_confidence` at which agreement stays at or above `min_agreement`, at that value and every higher one covering at least `min_samples` calls; `None` if no value qualifies |
| `at(min_confidence)` | a `Point` with `taken`, `agreed`, `total`, `coverage` and `agreement` |
| `curve(values=...)` | a list of `Point`s, by default every 0.05 from 0 to 1: ready to plot coverage and agreement against `min_confidence` |
| `bands(edges=...)` | calls grouped by confidence, with their agreement |

`suggest_min_confidence()` looks for agreement that *stays* above the bar from that point up,
so a lucky stretch can't pull it down, and it won't answer from a handful of calls: with too
little shadow traffic it returns `None`.

Agreement is not correctness: when the two disagree, the slow model is sometimes the one that's
wrong, and on ambiguous inputs either answer can be defensible. `Route.fast_output` shows what
the fast model would have said; look at a sample of the disagreements before deciding.

Small local decision models are much less sure of themselves than hosted ones: expect a lower
useful `min_confidence`, or most calls to stay on the slow side.
