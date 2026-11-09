"""Read an OpenAI-style request into a decision state and a JSON schema, or say why it can't be.

Only parameters that don't change what a correct answer is may be ignored. Anything else
(tools, streaming, audio, logprobs, stored conversations, extra request options) means the
caller relies on behaviour only the slow model has, so the request goes to it untouched.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from openai import NotGiven, Omit

from .schema import Ineligible

# Sampling, limits and bookkeeping: they shape how a model generates, not what the answer is.
_IGNORABLE = {
    "frequency_penalty",
    "max_completion_tokens",
    "max_output_tokens",
    "max_tokens",
    "metadata",
    "model",
    "presence_penalty",
    "prompt_cache_key",
    "prompt_cache_retention",
    "reasoning",
    "reasoning_effort",
    "safety_identifier",
    "seed",
    "service_tier",
    "store",
    "temperature",
    "timeout",
    "top_p",
    "truncation",
    "user",
    "verbosity",
}
_CHAT_INPUTS = {"messages", "response_format"}
_RESPONSES_INPUTS = {"input", "instructions", "text"}


@dataclass(frozen=True)
class Request:
    state: dict[str, Any]
    """``{"conversation": [{"role": ..., "content": ...}, ...]}``: structured, as System One
    servers recommend. Servers that accept only a string state can't be used as the fast model."""
    schema: Mapping[str, Any]
    name: str | None


def given(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Drop the SDK's ``omit``/``NOT_GIVEN`` placeholders."""
    return {k: v for k, v in kwargs.items() if not isinstance(v, (Omit, NotGiven))}


def chat_request(kwargs: Mapping[str, Any]) -> Request:
    kwargs = given(kwargs)
    _check_params(
        kwargs,
        _CHAT_INPUTS,
        falsy={"stream", "logprobs", "parallel_tool_calls"},
        n_ok=True,
    )
    response_format = kwargs.get("response_format")
    if not isinstance(response_format, Mapping) or response_format.get("type") != "json_schema":
        raise Ineligible("no json_schema response_format")
    json_schema = response_format.get("json_schema") or {}
    turns = [_chat_turn(message) for message in kwargs.get("messages") or ()]
    return _request(turns, json_schema.get("schema"), json_schema.get("name"))


def responses_request(kwargs: Mapping[str, Any]) -> Request:
    kwargs = given(kwargs)
    _check_params(kwargs, _RESPONSES_INPUTS, falsy={"stream", "background", "parallel_tool_calls"})
    text_format = (kwargs.get("text") or {}).get("format") or {}
    if text_format.get("type") != "json_schema":
        raise Ineligible("no json_schema text format")
    turns = []
    if instructions := kwargs.get("instructions"):
        turns.append({"role": "system", "content": _text(instructions)})
    match kwargs.get("input"):
        case str() as text:
            turns.append({"role": "user", "content": text})
        case None:
            pass
        case items:
            turns.extend(_responses_turn(item) for item in items)
    return _request(turns, text_format.get("schema"), text_format.get("name"))


def _check_params(
    kwargs: Mapping[str, Any], inputs: set[str], *, falsy: set[str], n_ok: bool = False
) -> None:
    for key, value in kwargs.items():
        if key in inputs or key in _IGNORABLE:
            continue
        if key in falsy and not value:
            continue
        if n_ok and key == "n" and value in (None, 1):
            continue
        if key in {"extra_headers", "extra_query", "extra_body"} and not value:
            continue
        raise Ineligible(f"uses {key}")


def _request(turns: list[dict[str, str]], schema: Any, name: str | None) -> Request:
    if not isinstance(schema, Mapping):
        raise Ineligible("json_schema has no schema")
    if not any(turn["role"] == "user" for turn in turns):
        raise Ineligible("no user message")
    return Request({"conversation": turns}, schema, name)


def _chat_turn(message: Any) -> dict[str, str]:
    if not isinstance(message, Mapping):
        raise Ineligible("message is not a dict")
    role = message.get("role")
    if role not in {"system", "developer", "user", "assistant"}:
        raise Ineligible(f"{role} message")
    if message.get("tool_calls") or message.get("function_call") or message.get("audio"):
        raise Ineligible("assistant tool call in history")
    return {
        "role": "system" if role == "developer" else role,
        "content": _text(message.get("content")),
    }


def _responses_turn(item: Any) -> dict[str, str]:
    if not isinstance(item, Mapping) or item.get("type", "message") != "message":
        raise Ineligible("non-message input item")
    role = item.get("role")
    if role not in {"system", "developer", "user", "assistant"}:
        raise Ineligible(f"{role} input")
    return {
        "role": "system" if role == "developer" else role,
        "content": _text(item.get("content")),
    }


def _text(content: Any) -> str:
    """Flatten text content; images, audio and files can't go to the fast model as plain text."""
    match content:
        case str():
            return content
        case None:
            return ""
        case Iterable():
            parts = []
            for part in content:
                if not isinstance(part, Mapping) or part.get("type") not in {
                    "text",
                    "input_text",
                    "output_text",
                }:
                    raise Ineligible("non-text content part")
                parts.append(part.get("text") or "")
            return "\n".join(parts)
        case _:
            raise Ineligible("unrecognized message content")
