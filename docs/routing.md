# Routing, monitoring and thresholds

## Route

After every intercepted call, `LazyConfig.on_route` receives a `Route`:

| Field | Meaning |
|---|---|
| `api` | `"chat.completions"` or `"responses"` |
| `answered_by` | `"fast"` or `"slow"` |
| `reason` | why; see below |
| `slow_model` | the model the call asked for |
| `fast_model` | the decision model that answered, when it was asked |
| `confidence` | the lowest field confidence, compared with the threshold |
| `field_confidences` | confidence per field, keyed by dotted path |
| `fast_input_tokens` | tokens the fast model billed |
| `detail` | why the request was ineligible, or the fast model's error |
| `shadow_mismatches` | in shadow mode, the fields where the two models disagreed |

| `reason` | `answered_by` | When |
|---|---|---|
| `confident` | fast | every field reached the threshold |
| `low_confidence` | slow | at least one field didn't |
| `ineligible` | slow | the schema or the request needs the slow model ([details](schemas.md#what-sends-a-request-to-the-slow-model)) |
| `fast_model_error` | slow | the fast model failed, or its answer was rejected (for example by a validator in your Pydantic model) |
| `disabled` | slow | `enabled=False`, or no key for the fast model |
| `shadow` | slow | shadow mode |

## Watching the traffic

`on_route` is any callable; a list's `append` collects routes for inspection.

```python
from collections import Counter
from lazycontroller import LazyConfig, OpenAI, Route

seen = Counter()


def track(route: Route) -> None:
    seen[route.answered_by, route.reason] += 1
    if route.reason == "fast_model_error":
        logger.warning("fast model failed: %s", route.detail)


client = OpenAI(lazy=LazyConfig(on_route=track))
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

`LazyConfig(shadow=True)` asks both models, always returns the slow model's answer, and fills
`Route.shadow_mismatches` with the fields where the two disagreed. Nothing changes for your
users, and you get the data to decide whether, and at which threshold, to let the fast model
answer.

## Choosing a threshold

The default, 0.9, is cautious. The right value depends on the decision model and on what a
wrong answer costs you.

1. Run in shadow mode on real traffic for a while, collecting routes.
2. For each candidate threshold, count how many calls the fast model would have taken
   (`confidence >= t`) and how often it agreed with the slow model on those.
3. Pick the lowest threshold whose disagreement rate you can live with, and turn shadow off.
   Use `with_lazy` to keep a higher threshold where mistakes are expensive.

```python
scored = [r for r in routes if r.reason == "shadow"]
for t in (0.5, 0.6, 0.7, 0.8, 0.9):
    taken = [r for r in scored if r.confidence >= t]
    agreed = sum(not r.shadow_mismatches for r in taken)
    print(f"{t}: fast takes {len(taken)}/{len(scored)}, agrees on {agreed}/{len(taken)}")
```

Agreement is not correctness: when the two disagree, the slow model is sometimes the one that's
wrong. Look at a sample of the disagreements before deciding.

Small local decision models are less sure of themselves than large hosted ones: with
`kev:0.8b-q8_0`, 0.7–0.8 was the useful range in our tests, while 0.9 let almost nothing
through.
