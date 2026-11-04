"""Fakes for both sides: a scripted Jev client and an OpenAI HTTP transport that records calls."""

import json
from typing import Any

import httpx2
import pytest
from typesafe_sdk import (
    Choice,
    Noul,
    Score,
    SystemOneResponse,
    TypeSafeAPIConnectionError,
)

import lazycontroller


class FakeJev:
    """Answers each question as confidently as ``confidence`` says, picking ``picks[field]``.

    Fields are matched by the dotted path at the end of their question instructions'
    backticked name, so tests can talk about ``priority`` instead of question ids.
    """

    def __init__(
        self,
        picks: dict[str, Any] | None = None,
        confidence: float = 0.98,
        error: bool = False,
    ):
        self.picks = picks or {}
        self.confidence = confidence
        self.error = error
        self.requests: list[dict[str, Any]] = []

    def system_one(self, *, state, questions, model=None, **_):
        self.requests.append({"state": state, "questions": questions, "model": model})
        if self.error:
            raise TypeSafeAPIConnectionError("connection refused")
        answers = {qid: self._answer(q) for qid, q in questions.items()}
        return SystemOneResponse.model_validate(
            {
                "model": "jev-1.13.0",
                "usage": {"input_tokens": 120, "output_tokens": 9},
                "answers": answers,
            }
        )

    def _answer(self, question):
        field = question.instructions.split("`")[1]
        pick = self.picks.get(field)
        c = self.confidence
        match question:
            case Choice():
                labels = list(question.criteria)
                choice = pick if pick is not None else labels[0]
                rest = (1 - c) / max(len(labels) - 1, 1)
                probs = {label: (c if label == choice else rest) for label in labels}
                return {
                    "type": "choice",
                    "choice": choice,
                    "confidence": c,
                    "probabilities": probs,
                }
            case Noul():
                p = (1 + c) / 2 if pick is not False else (1 - c) / 2
                return {"type": "noul", "noul": p}
            case Score():
                level = pick if pick is not None else 0
                n = len(question.criteria)
                probs = {i: (c if i == level else (1 - c) / (n - 1)) for i in range(n)}
                legend = dict(enumerate(question.criteria))
                return {
                    "type": "score",
                    "score": float(level),
                    "confidence": c,
                    "legend": legend,
                    "probabilities": probs,
                }
        raise AssertionError(question)


class AsyncFakeJev(FakeJev):
    async def system_one(self, **kwargs):
        return FakeJev.system_one(self, **kwargs)


class FakeOpenAI:
    """An httpx2 transport that answers OpenAI endpoints with ``content`` and records requests."""

    def __init__(self, content: dict[str, Any]):
        self.content = json.dumps(content)
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        self.requests.append({"path": request.url.path, "body": body})
        if request.url.path.endswith("/chat/completions"):
            return httpx2.Response(200, json=_chat_completion(body["model"], self.content))
        if request.url.path.endswith("/responses"):
            return httpx2.Response(200, json=_response(body["model"], self.content))
        return httpx2.Response(404, json={"error": {"message": "not faked"}})


def _chat_completion(model: str, content: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
    }


def _response(model: str, content: str) -> dict[str, Any]:
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": model,
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": content, "annotations": []}],
            }
        ],
        "tools": [],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
    }


@pytest.fixture
def routes():
    return []


@pytest.fixture
def make_client(routes):
    """Build a lazycontroller.OpenAI (or AsyncOpenAI) wired to a fake Jev and fake OpenAI."""

    def make(jev=None, openai_content=None, *, is_async=False, **config):
        fake_openai = FakeOpenAI(openai_content or {})
        lazy = lazycontroller.LazyConfig(fast_client=jev, on_route=routes.append, **config)
        if is_async:
            http = httpx2.AsyncClient(transport=httpx2.MockTransport(fake_openai))
            client = lazycontroller.AsyncOpenAI(api_key="sk-test", http_client=http, lazy=lazy)
        else:
            http = httpx2.Client(transport=httpx2.MockTransport(fake_openai))
            client = lazycontroller.OpenAI(api_key="sk-test", http_client=http, lazy=lazy)
        return client, fake_openai

    return make
