"""Decide, per request, whether the fast model answers or the call goes on to the slow one.

The fast model is a decision model on a System One API: Jev on TypeSafe, a local Kev on
Ollama, or the same models hosted elsewhere. The slow model is whatever the caller's
OpenAI-compatible client was going to call anyway.
"""

import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Self

from typesafe_sdk import (
    AsyncTypeSafeClient,
    SystemOneResponse,
    TypeSafeClient,
)

from .requests import Request
from .schema import Ineligible, Plan, compile_schema

logger = logging.getLogger("lazycontroller")

type Api = Literal["chat.completions", "responses"]
type Reason = Literal[
    "confident", "low_confidence", "ineligible", "fast_model_error", "disabled", "shadow"
]


@dataclass(frozen=True)
class Route:
    """How one request was answered; passed to ``LazyConfig.on_route``."""

    api: Api
    answered_by: Literal["fast", "slow"]
    reason: Reason
    """Why: the fast model was ``confident``, or the slow model was called because of
    ``low_confidence``, an ``ineligible`` request, a ``fast_model_error``, routing being
    ``disabled``, or ``shadow`` mode."""
    slow_model: str | None
    """The model the caller asked for."""
    fast_model: str | None = None
    confidence: float | None = None
    """The lowest field confidence, which is what the threshold is compared with."""
    field_confidences: Mapping[str, float] = field(default_factory=dict)
    fast_output: Mapping[str, Any] | None = None
    """The fast model's answer as a JSON object, whenever it was asked (also when the slow model
    answered in the end, so shadow disagreements can be inspected)."""
    fast_input_tokens: int | None = None
    detail: str | None = None
    """Why the request was ineligible, or the fast model's error."""
    shadow_mismatches: tuple[str, ...] | None = None
    """In shadow mode, the fields where the two models disagreed (empty when they agree)."""


@dataclass(frozen=True)
class LazyConfig:
    """Settings for routing structured-output requests to a fast decision model."""

    enabled: bool = True
    """``False`` sends every call straight to the slow model."""
    threshold: float = 0.9
    """Every field's confidence must reach this for the fast model's answer to be used."""
    fast_base_url: str | None = None
    """The System One server: ``https://api.typesafe.ai``, ``https://openrouter.ai/api``,
    ``http://localhost:11434`` for Ollama, ... Without it (or ``fast_client``) the fast side is
    off and every call goes to the slow model."""
    fast_api_key: str | None = None
    """Key for ``fast_base_url``, and sent nowhere else. Local servers need none."""
    fast_model: str | None = None
    """The decision model; ``None`` means the server's default."""
    shadow: bool = False
    """Ask the fast model but always return the slow model's answer, reporting where they
    differ."""
    on_route: Callable[[Route], None] | None = None
    fast_client: TypeSafeClient | AsyncTypeSafeClient | None = None
    """Use this System One client instead of creating one."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")


class _Router[C]:
    def __init__(self, config: LazyConfig, client_cls: type[C]) -> None:
        self.config = config
        self._client_cls = client_cls
        self._fast: C | None = config.fast_client  # type: ignore[assignment]

    @property
    def fast(self) -> C | None:
        if self._fast is None:
            if base_url := self.config.fast_base_url:
                # Passed explicitly, so the SDK reads neither TYPESAFE_API_KEY nor
                # TYPESAFE_BASE_URL. It insists on some key; local servers ignore it.
                api_key = self.config.fast_api_key or "local"
                self._fast = self._client_cls(api_key=api_key, base_url=base_url)
        return self._fast

    def derive(self, **changes: Any) -> Self:
        """A router with some settings changed, sharing the fast model's connection when it can."""
        config = replace(self.config, **changes)
        router = type(self)(config)
        if all(
            getattr(config, name) == getattr(self.config, name)
            for name in ("fast_api_key", "fast_base_url", "fast_client")
        ):
            router._fast = self._fast
        return router

    def plan(
        self, api: Api, prepare: Callable[[], Request], model: str | None
    ) -> tuple[Request, Plan] | Route:
        if not self.config.enabled:
            return Route(api, "slow", "disabled", model, detail="disabled in LazyConfig")
        try:
            if self.fast is None:
                return Route(api, "slow", "disabled", model, detail="no fast_base_url")
            request = prepare()
            return request, compile_schema(request.schema, title=request.name)
        except Ineligible as error:
            return Route(api, "slow", "ineligible", model, detail=str(error))
        except Exception as error:
            # A request we can't read is the slow model's to accept or reject, not ours.
            logger.warning("Could not read request, passing it to the slow model: %r", error)
            detail = f"could not read request: {type(error).__name__}: {error}"
            return Route(api, "slow", "ineligible", model, detail=detail)

    def judge(
        self, api: Api, plan: Plan, answer: SystemOneResponse, model: str | None
    ) -> tuple[dict[str, Any], Route]:
        output, confidences = plan.assemble(answer.answers)
        confidence = min(confidences.values())
        confident = confidence >= self.config.threshold
        route = Route(
            api,
            "fast" if confident and not self.config.shadow else "slow",
            "shadow" if self.config.shadow else "confident" if confident else "low_confidence",
            model,
            fast_model=answer.model,
            confidence=confidence,
            field_confidences=confidences,
            fast_output=output,
            fast_input_tokens=answer.usage.input_tokens,
        )
        return output, route

    def fast_failed(self, api: Api, model: str | None, error: Exception) -> Route:
        logger.warning("Fast model failed, falling back to the slow model: %r", error)
        detail = f"{type(error).__name__}: {error}"
        return Route(api, "slow", "fast_model_error", model, detail=detail)

    def report(self, route: Route) -> None:
        if self.config.on_route is not None:
            self.config.on_route(route)


@dataclass(frozen=True)
class Call:
    """One intercepted SDK call: how to read it, how to answer it, how to pass it on."""

    api: Api
    model: str | None
    prepare: Callable[[], Request]
    build: Callable[[dict[str, Any], SystemOneResponse], Any]
    output_json: Callable[[Any], Any]
    """Extract the JSON object from a slow-model response, for shadow comparisons."""


class Router(_Router[TypeSafeClient]):
    def __init__(self, config: LazyConfig) -> None:
        super().__init__(config, TypeSafeClient)

    def run(self, call: Call, slow: Callable[[], Any]) -> Any:
        planned = self.plan(call.api, call.prepare, call.model)
        if isinstance(planned, Route):
            self.report(planned)
            return slow()
        request, plan = planned
        try:
            answer = self.fast.system_one(  # type: ignore[union-attr]
                state=request.state, questions=plan.questions, model=self.config.fast_model
            )
            output, route = self.judge(call.api, plan, answer, call.model)
        except Exception as error:
            self.report(self.fast_failed(call.api, call.model, error))
            return slow()
        if route.answered_by == "fast":
            try:
                result = call.build(output, answer)
            except Exception as error:  # e.g. a validator in the caller's model rejects it
                self.report(self.fast_failed(call.api, call.model, error))
                return slow()
            self.report(route)
            return result
        result = slow()
        self.report(_with_shadow(route, plan, output, call, result))
        return result


class AsyncRouter(_Router[AsyncTypeSafeClient]):
    def __init__(self, config: LazyConfig) -> None:
        super().__init__(config, AsyncTypeSafeClient)

    async def run(self, call: Call, slow: Callable[[], Awaitable[Any]]) -> Any:
        planned = self.plan(call.api, call.prepare, call.model)
        if isinstance(planned, Route):
            self.report(planned)
            return await slow()
        request, plan = planned
        try:
            answer = await self.fast.system_one(  # type: ignore[union-attr]
                state=request.state, questions=plan.questions, model=self.config.fast_model
            )
            output, route = self.judge(call.api, plan, answer, call.model)
        except Exception as error:
            self.report(self.fast_failed(call.api, call.model, error))
            return await slow()
        if route.answered_by == "fast":
            try:
                result = call.build(output, answer)
            except Exception as error:  # e.g. a validator in the caller's model rejects it
                self.report(self.fast_failed(call.api, call.model, error))
                return await slow()
            self.report(route)
            return result
        result = await slow()
        self.report(_with_shadow(route, plan, output, call, result))
        return result


def _with_shadow(
    route: Route, plan: Plan, output: dict[str, Any], call: Call, result: Any
) -> Route:
    if route.reason != "shadow":
        return route
    try:
        theirs = call.output_json(result)
    except (TypeError, ValueError, AttributeError, IndexError):
        theirs = None
    mismatches = tuple(
        ".".join(f.path) for f in plan.fields if _get(output, f.path) != _get(theirs, f.path)
    )
    return Route(**{**route.__dict__, "shadow_mismatches": mismatches})


def _get(obj: Any, path: tuple[str, ...]) -> Any:
    for key in path:
        if not isinstance(obj, Mapping) or key not in obj:
            return _MISSING
        obj = obj[key]
    return obj


_MISSING = object()


def chat_output_json(completion: Any) -> Any:
    return json.loads(completion.choices[0].message.content)


def responses_output_json(response: Any) -> Any:
    return json.loads(response.output_text)
