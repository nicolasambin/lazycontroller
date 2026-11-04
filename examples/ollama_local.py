"""Try lazycontroller fully offline: a decision model and a "slow" LLM, both served by Ollama.

    ollama pull kev:0.8b-q8_0 && ollama pull qwen3.5:4b
    uv run examples/ollama_local.py [--choice-only] [--think] [--decision MODEL] [--slow MODEL]

Runs a few support tickets in shadow mode (both models answer, the slow one's answer is
returned) and shows, for each threshold, how many tickets the decision model would have taken
and how often it agreed with the slow model on those.
"""

import argparse
import time
from typing import Literal

from pydantic import BaseModel, Field

from lazycontroller import LazyConfig, OpenAI, Route

OLLAMA = "http://localhost:11434"

TICKETS = [
    "You charged my card twice for the March invoice. I want a refund today.",
    "Every API call has returned 500 for the last 20 minutes and our checkout is down!!!",
    "Hi, could you tell me if the Team plan includes SSO? Thinking of upgrading next quarter.",
    "The export button does nothing in Firefox. Not urgent, Chrome works fine.",
    "I've emailed three times about my cancelled subscription still being billed. Unacceptable.",
    "How do I add a second admin to our account?",
    "Your webhook signatures stopped validating after yesterday's release, payments are failing.",
    "Is there a discount for nonprofits?",
]


class Ticket(BaseModel):
    team: Literal["billing", "technical", "sales"] = Field(
        description="Which team should handle it"
    )
    is_urgent: bool = Field(description="The customer needs this solved today")
    frustration: int = Field(ge=1, le=3, description="1 calm, 2 annoyed, 3 angry")


class TeamOnly(BaseModel):
    team: Literal["billing", "technical", "sales"] = Field(
        description="Which team should handle it"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision", default="kev:0.8b-q8_0")
    parser.add_argument("--slow", default="qwen3.5:4b")
    parser.add_argument("--choice-only", action="store_true", help="ask only for the team")
    parser.add_argument("--think", action="store_true", help="let the slow model reason first")
    args = parser.parse_args()
    schema = TeamOnly if args.choice_only else Ticket

    routes: list[Route] = []
    client = OpenAI(
        base_url=f"{OLLAMA}/v1",
        lazy=LazyConfig(
            fast_model=args.decision, fast_base_url=OLLAMA, shadow=True, on_route=routes.append
        ),
    )

    print(
        f"decision model: {args.decision}   slow model: {args.slow}   schema: {schema.__name__}\n"
    )
    for text in TICKETS:
        start = time.perf_counter()
        completion = client.chat.completions.parse(
            model=args.slow,
            messages=[
                {"role": "system", "content": "Triage this customer support ticket."},
                {"role": "user", "content": text},
            ],
            response_format=schema,
            reasoning_effort="medium" if args.think else "none",
        )
        elapsed = time.perf_counter() - start
        route = routes[-1]
        slow = completion.choices[0].message.parsed
        if route.reason != "shadow":
            print(f"[{route.reason}: {route.detail}] {text[:60]}")
            continue
        verdict = (
            "agree"
            if not route.shadow_mismatches
            else "differ on " + ", ".join(route.shadow_mismatches)
        )
        print(f"{text[:60]:<62} conf {route.confidence:.2f}  {elapsed:5.1f}s  {verdict}")
        print(f"{'':<62} slow: {slow.model_dump()}")

    scored = [r for r in routes if r.reason == "shadow"]
    print("\nthreshold  taken by decision model  agreement on those")
    for threshold in (0.0, 0.5, 0.7, 0.8, 0.9):
        taken = [r for r in scored if r.confidence >= threshold]
        agreed = sum(1 for r in taken if not r.shadow_mismatches)
        rate = f"{agreed}/{len(taken)}" if taken else "-"
        print(f"{threshold:>9.1f}  {len(taken):>3}/{len(scored):<20} {rate}")


if __name__ == "__main__":
    main()
