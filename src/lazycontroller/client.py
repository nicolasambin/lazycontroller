"""``OpenAI`` and ``AsyncOpenAI`` subclasses that let a fast model answer structured-output calls.

Intercepted: ``chat.completions.create`` / ``parse`` and ``responses.create`` / ``parse``.
Everything else is the stock SDK, and so is every intercepted call the fast model can't take.
"""

import os
from functools import cached_property
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import openai
from openai.lib._parsing import parse_chat_completion, type_to_response_format_param
from openai.lib._parsing._responses import parse_response, type_to_text_format_param
from openai.resources.chat import AsyncChat, AsyncCompletions, Chat, Completions
from openai.resources.responses import AsyncResponses, Responses

from . import build
from .lazy import (
    AsyncRouter,
    Call,
    LazyConfig,
    Router,
    chat_output_json,
    responses_output_json,
)
from .requests import chat_request, given, responses_request
from .schema import Ineligible


def _chat_call(kwargs: dict[str, Any]) -> Call:
    return Call(
        "chat.completions",
        kwargs.get("model"),
        lambda: chat_request(kwargs),
        lambda output, answer: build.chat_completion(output, answer),
        chat_output_json,
    )


def _chat_parse_call(kwargs: dict[str, Any]) -> Call:
    response_format = kwargs.get("response_format")

    def prepare():
        if not isinstance(response_format, type):
            raise Ineligible("response_format is not a type")
        return chat_request(
            {
                **kwargs,
                "response_format": type_to_response_format_param(response_format),
            }
        )

    def make(output, answer):
        return parse_chat_completion(
            response_format=response_format,
            input_tools=kwargs.get("tools", openai.omit),
            chat_completion=build.chat_completion(output, answer),
        )

    return Call("chat.completions", kwargs.get("model"), prepare, make, chat_output_json)


def _responses_call(kwargs: dict[str, Any]) -> Call:
    return Call(
        "responses",
        kwargs.get("model"),
        lambda: responses_request(kwargs),
        lambda output, answer: build.response(output, answer, given(kwargs)),
        responses_output_json,
    )


def _responses_parse_call(kwargs: dict[str, Any]) -> Call:
    text_format = kwargs.get("text_format")

    def with_format() -> dict[str, Any]:
        if not isinstance(text_format, type):
            raise Ineligible("text_format is not a type")
        if "parse_as" in given(kwargs):
            raise Ineligible("uses parse_as")
        rest = {k: v for k, v in kwargs.items() if k != "text_format"}
        return {
            **rest,
            "text": {
                **(rest.get("text") or {}),
                "format": type_to_text_format_param(text_format),
            },
        }

    def make(output, answer):
        return parse_response(
            text_format=text_format,
            input_tools=kwargs.get("tools"),
            response=build.response(output, answer, given(with_format())),
        )

    return Call(
        "responses",
        kwargs.get("model"),
        lambda: responses_request(with_format()),
        make,
        responses_output_json,
    )


class _Completions(Completions):
    # The overrides take **kwargs; type checkers keep seeing the SDK's full signatures.
    if not TYPE_CHECKING:

        def create(self, **kwargs):
            return self._client.lazy.run(
                _chat_call(kwargs), lambda: super(_Completions, self).create(**kwargs)
            )

        def parse(self, **kwargs):
            return self._client.lazy.run(
                _chat_parse_call(kwargs),
                lambda: super(_Completions, self).parse(**kwargs),
            )


class _AsyncCompletions(AsyncCompletions):
    if not TYPE_CHECKING:

        async def create(self, **kwargs):
            return await self._client.lazy.run(
                _chat_call(kwargs),
                lambda: super(_AsyncCompletions, self).create(**kwargs),
            )

        async def parse(self, **kwargs):
            return await self._client.lazy.run(
                _chat_parse_call(kwargs),
                lambda: super(_AsyncCompletions, self).parse(**kwargs),
            )


class _Responses(Responses):
    if not TYPE_CHECKING:

        def create(self, **kwargs):
            return self._client.lazy.run(
                _responses_call(kwargs),
                lambda: super(_Responses, self).create(**kwargs),
            )

        def parse(self, **kwargs):
            return self._client.lazy.run(
                _responses_parse_call(kwargs),
                lambda: super(_Responses, self).parse(**kwargs),
            )


class _AsyncResponses(AsyncResponses):
    if not TYPE_CHECKING:

        async def create(self, **kwargs):
            return await self._client.lazy.run(
                _responses_call(kwargs),
                lambda: super(_AsyncResponses, self).create(**kwargs),
            )

        async def parse(self, **kwargs):
            return await self._client.lazy.run(
                _responses_parse_call(kwargs),
                lambda: super(_AsyncResponses, self).parse(**kwargs),
            )


class _Chat(Chat):
    @cached_property
    def completions(self) -> Completions:
        return _Completions(self._client)


class _AsyncChat(AsyncChat):
    @cached_property
    def completions(self) -> AsyncCompletions:
        return _AsyncCompletions(self._client)


_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
_CREDENTIALS = ("api_key", "admin_api_key", "workload_identity")


def _with_local_key(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Fill in a placeholder key for a local server (Ollama, vLLM, LM Studio...).

    The SDK refuses to start without credentials, but local servers ignore them. Only applies
    where ``openai.OpenAI`` would have raised, so nothing changes for a configured client.
    """
    if any(kwargs.get(name) is not None for name in _CREDENTIALS):
        return kwargs
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_ADMIN_KEY"):
        return kwargs
    base_url = kwargs.get("base_url") or os.environ.get("OPENAI_BASE_URL")
    host = urlsplit(str(base_url)).hostname if base_url else None
    if host in _LOCAL_HOSTS or (host or "").endswith(".localhost"):
        return {**kwargs, "api_key": "local"}
    return kwargs


class OpenAI(openai.OpenAI):
    """``openai.OpenAI`` that lets a fast model answer the structured-output calls it is sure of.

    Takes every ``openai.OpenAI`` argument, plus ``lazy`` for the routing settings. A local
    ``base_url`` such as Ollama's needs no ``api_key``.
    """

    lazy: Router

    def __init__(self, *args: Any, lazy: LazyConfig | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **_with_local_key(kwargs))
        self.lazy = Router(lazy or LazyConfig())

    @cached_property
    def chat(self) -> Chat:
        return _Chat(self)

    @cached_property
    def responses(self) -> Responses:
        return _Responses(self)

    def copy(self, *args: Any, **kwargs: Any) -> "OpenAI":
        new = super().copy(*args, **kwargs)
        new.lazy = self.lazy
        return new

    with_options = copy

    def with_lazy(self, **changes: Any) -> "OpenAI":
        """A copy of this client with some ``LazyConfig`` settings changed.

        The ``with_options`` of routing: ``client.with_lazy(threshold=0.95)`` for calls where a
        wrong answer costs more, ``client.with_lazy(enabled=False)`` to skip the fast model.
        The original client is unchanged.
        """
        new = self.copy()
        new.lazy = self.lazy.derive(**changes)
        return new


class AsyncOpenAI(openai.AsyncOpenAI):
    """``openai.AsyncOpenAI`` counterpart of ``lazycontroller.OpenAI``."""

    lazy: AsyncRouter

    def __init__(self, *args: Any, lazy: LazyConfig | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **_with_local_key(kwargs))
        self.lazy = AsyncRouter(lazy or LazyConfig())

    @cached_property
    def chat(self) -> AsyncChat:
        return _AsyncChat(self)

    @cached_property
    def responses(self) -> AsyncResponses:
        return _AsyncResponses(self)

    def copy(self, *args: Any, **kwargs: Any) -> "AsyncOpenAI":
        new = super().copy(*args, **kwargs)
        new.lazy = self.lazy
        return new

    with_options = copy

    def with_lazy(self, **changes: Any) -> "AsyncOpenAI":
        """A copy of this client with some ``LazyConfig`` settings changed.

        The ``with_options`` of routing: ``client.with_lazy(threshold=0.95)`` for calls where a
        wrong answer costs more, ``client.with_lazy(enabled=False)`` to skip the fast model.
        The original client is unchanged.
        """
        new = self.copy()
        new.lazy = self.lazy.derive(**changes)
        return new
