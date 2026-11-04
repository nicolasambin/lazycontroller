import json
from typing import Literal

import openai
import pytest
from conftest import AsyncFakeJev, FakeJev
from openai.types.chat import ParsedChatCompletion
from pydantic import BaseModel, Field

import lazycontroller


class Ticket(BaseModel):
    team: Literal["billing", "technical", "sales"]
    is_urgent: bool
    severity: int = Field(ge=1, le=3)


class TicketWithSummary(Ticket):
    summary: str


MESSAGES = [
    {"role": "system", "content": "Triage the support ticket."},
    {"role": "user", "content": "Every API call returns 500 since 20 minutes ago!"},
]
JEV_PICKS = {"team": "technical", "is_urgent": True, "severity": 2}
OPENAI_TICKET = {"team": "billing", "is_urgent": False, "severity": 1}


def test_drop_in_subclass_of_openai_client():
    client = lazycontroller.OpenAI(api_key="sk-test")
    assert isinstance(client, openai.OpenAI)
    assert client.with_options(timeout=5).lazy is client.lazy


def test_confident_fast_model_answers_chat_parse_alone(make_client, routes):
    jev = FakeJev(JEV_PICKS)
    client, fake_openai = make_client(jev, OPENAI_TICKET)

    completion = client.chat.completions.parse(
        model="gpt-5", messages=MESSAGES, response_format=Ticket
    )

    assert fake_openai.requests == []
    assert isinstance(completion, ParsedChatCompletion)
    assert completion.model == "jev-1.13.0"
    assert completion.choices[0].message.parsed == Ticket(
        team="technical", is_urgent=True, severity=3
    )
    assert json.loads(completion.choices[0].message.content) == {
        "team": "technical",
        "is_urgent": True,
        "severity": 3,
    }
    assert completion.usage.prompt_tokens == 120

    route = routes[0]
    assert (route.answered_by, route.reason, route.slow_model) == (
        "fast",
        "confident",
        "gpt-5",
    )
    assert route.confidence == pytest.approx(0.98)
    assert set(route.field_confidences) == {"team", "is_urgent", "severity"}

    state = jev.requests[0]["state"]
    assert state == {
        "conversation": [
            {"role": "system", "content": "Triage the support ticket."},
            MESSAGES[1],
        ]
    }


def test_low_confidence_falls_back_to_the_requested_model(make_client, routes):
    client, fake_openai = make_client(FakeJev(JEV_PICKS, confidence=0.6), OPENAI_TICKET)

    completion = client.chat.completions.parse(
        model="gpt-5", messages=MESSAGES, response_format=Ticket
    )

    assert [r["body"]["model"] for r in fake_openai.requests] == ["gpt-5"]
    assert completion.choices[0].message.parsed == Ticket(**OPENAI_TICKET)
    assert routes[0].reason == "low_confidence"
    assert routes[0].confidence == pytest.approx(0.6)


def test_threshold_is_configurable(make_client):
    client, fake_openai = make_client(
        FakeJev(JEV_PICKS, confidence=0.6), OPENAI_TICKET, threshold=0.15
    )
    client.chat.completions.parse(model="gpt-5", messages=MESSAGES, response_format=Ticket)
    assert fake_openai.requests == []


def test_free_text_field_goes_to_openai(make_client, routes):
    jev = FakeJev(JEV_PICKS)
    client, fake_openai = make_client(jev, {**OPENAI_TICKET, "summary": "API down"})

    completion = client.chat.completions.parse(
        model="gpt-5", messages=MESSAGES, response_format=TicketWithSummary
    )

    assert jev.requests == []
    assert len(fake_openai.requests) == 1
    assert completion.choices[0].message.parsed.summary == "API down"
    assert (routes[0].reason, routes[0].detail) == (
        "ineligible",
        "summary: free-text string",
    )


def test_chat_create_with_json_schema_dict(make_client, routes):
    client, fake_openai = make_client(FakeJev(JEV_PICKS), OPENAI_TICKET)
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "ticket",
            "strict": True,
            "schema": openai.pydantic_function_tool(Ticket)["function"]["parameters"],
        },
    }

    completion = client.chat.completions.create(
        model="gpt-5", messages=MESSAGES, response_format=response_format
    )

    assert fake_openai.requests == []
    assert json.loads(completion.choices[0].message.content)["team"] == "technical"
    assert routes[0].answered_by == "fast"


@pytest.mark.parametrize(
    ("extra", "detail"),
    [
        ({"tools": [openai.pydantic_function_tool(Ticket)]}, "uses tools"),
        ({"logprobs": True}, "uses logprobs"),
        ({"n": 2}, "uses n"),
        ({"extra_body": {"x": 1}}, "uses extra_body"),
    ],
)
def test_options_the_fast_model_cannot_honour_go_to_slow_model(make_client, routes, extra, detail):
    client, fake_openai = make_client(FakeJev(JEV_PICKS), OPENAI_TICKET)
    client.chat.completions.parse(model="gpt-5", messages=MESSAGES, response_format=Ticket, **extra)
    assert len(fake_openai.requests) == 1
    assert routes[0].detail == detail


def test_images_go_to_openai(make_client, routes):
    client, fake_openai = make_client(FakeJev(JEV_PICKS), OPENAI_TICKET)
    messages = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}],
        }
    ]
    client.chat.completions.parse(model="gpt-5", messages=messages, response_format=Ticket)
    assert len(fake_openai.requests) == 1
    assert routes[0].detail == "non-text content part"


def test_raw_response_bypasses_the_fast_model(make_client):
    client, fake_openai = make_client(FakeJev(JEV_PICKS), OPENAI_TICKET)
    raw = client.chat.completions.with_raw_response.parse(
        model="gpt-5", messages=MESSAGES, response_format=Ticket
    )
    assert raw.http_response.status_code == 200
    assert len(fake_openai.requests) == 1


def test_plain_chat_without_schema_is_untouched(make_client, routes):
    jev = FakeJev()
    client, fake_openai = make_client(jev, OPENAI_TICKET)
    client.chat.completions.create(model="gpt-5", messages=MESSAGES)
    assert jev.requests == []
    assert len(fake_openai.requests) == 1
    assert routes[0].reason == "ineligible"


def test_no_typesafe_key_means_plain_openai(make_client, routes, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    client, fake_openai = make_client(None, OPENAI_TICKET)
    client.chat.completions.parse(model="gpt-5", messages=MESSAGES, response_format=Ticket)
    assert len(fake_openai.requests) == 1
    assert routes[0].reason == "disabled"


def test_fast_model_errors_fall_back_to_slow_model(make_client, routes):
    client, fake_openai = make_client(FakeJev(error=True), OPENAI_TICKET)
    completion = client.chat.completions.parse(
        model="gpt-5", messages=MESSAGES, response_format=Ticket
    )
    assert completion.choices[0].message.parsed == Ticket(**OPENAI_TICKET)
    assert routes[0].reason == "fast_model_error"


def test_shadow_mode_returns_openai_and_reports_disagreements(make_client, routes):
    client, fake_openai = make_client(
        FakeJev(JEV_PICKS),
        {"team": "technical", "is_urgent": False, "severity": 3},
        shadow=True,
    )

    completion = client.chat.completions.parse(
        model="gpt-5", messages=MESSAGES, response_format=Ticket
    )

    assert len(fake_openai.requests) == 1
    assert completion.model == "gpt-5"
    route = routes[0]
    assert (route.answered_by, route.reason) == ("slow", "shadow")
    assert route.shadow_mismatches == ("is_urgent",)


def test_responses_parse_answered_by_fast_model(make_client, routes):
    jev = FakeJev(JEV_PICKS)
    client, fake_openai = make_client(jev, OPENAI_TICKET)

    response = client.responses.parse(
        model="gpt-5",
        instructions="Triage the support ticket.",
        input="Every API call returns 500 since 20 minutes ago!",
        text_format=Ticket,
    )

    assert fake_openai.requests == []
    assert response.output_parsed == Ticket(team="technical", is_urgent=True, severity=3)
    assert response.model == "jev-1.13.0"
    assert jev.requests[0]["state"]["conversation"][0] == {
        "role": "system",
        "content": "Triage the support ticket.",
    }
    assert routes[0].api == "responses"


def test_responses_create_with_message_items_falls_back_on_low_confidence(make_client, routes):
    client, fake_openai = make_client(FakeJev(JEV_PICKS, confidence=0.5), OPENAI_TICKET)
    text = {
        "format": {
            "type": "json_schema",
            "name": "ticket",
            "schema": openai.pydantic_function_tool(Ticket)["function"]["parameters"],
        }
    }

    response = client.responses.create(
        model="gpt-5",
        input=[{"role": "user", "content": [{"type": "input_text", "text": "API down"}]}],
        text=text,
    )

    assert json.loads(response.output_text) == OPENAI_TICKET
    assert fake_openai.requests[0]["path"].endswith("/responses")
    assert routes[0].reason == "low_confidence"


async def test_async_client_routes_the_same_way(make_client, routes):
    client, fake_openai = make_client(AsyncFakeJev(JEV_PICKS), OPENAI_TICKET, is_async=True)

    confident = await client.chat.completions.parse(
        model="gpt-5", messages=MESSAGES, response_format=Ticket
    )
    assert confident.choices[0].message.parsed.team == "technical"
    assert fake_openai.requests == []

    low, _ = make_client(AsyncFakeJev(JEV_PICKS, confidence=0.5), OPENAI_TICKET, is_async=True)
    response = await low.responses.parse(model="gpt-5", input="API down", text_format=Ticket)
    assert response.output_parsed == Ticket(**OPENAI_TICKET)
    assert [r.reason for r in routes] == ["confident", "low_confidence"]


@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:11434/v1",
        "http://127.0.0.1:8000/v1",
        "http://[::1]:1234/v1",
        "http://ollama.localhost/v1",
    ],
)
def test_local_server_needs_no_api_key(monkeypatch, base_url):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_ADMIN_KEY", raising=False)
    assert lazycontroller.OpenAI(base_url=base_url).api_key == "local"
    assert lazycontroller.AsyncOpenAI(base_url=base_url).api_key == "local"


def test_local_base_url_from_env_needs_no_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_ADMIN_KEY", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    assert lazycontroller.OpenAI().api_key == "local"


def test_remote_server_still_requires_a_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_ADMIN_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    with pytest.raises(openai.OpenAIError):
        lazycontroller.OpenAI()
    with pytest.raises(openai.OpenAIError):
        lazycontroller.OpenAI(base_url="https://localhost.example.com/v1")


def test_explicit_or_env_key_is_kept(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    assert lazycontroller.OpenAI(base_url="http://localhost:11434/v1").api_key == "sk-env"
    assert (
        lazycontroller.OpenAI(base_url="http://localhost:11434/v1", api_key="sk-x").api_key
        == "sk-x"
    )


def test_with_lazy_changes_only_the_copy(make_client, routes):
    jev = FakeJev(JEV_PICKS, confidence=0.6)
    client, fake_openai = make_client(jev, OPENAI_TICKET)
    lenient = client.with_lazy(threshold=0.5)

    lenient.chat.completions.parse(model="gpt-5", messages=MESSAGES, response_format=Ticket)
    client.chat.completions.parse(model="gpt-5", messages=MESSAGES, response_format=Ticket)

    assert [r.reason for r in routes] == ["confident", "low_confidence"]
    assert len(fake_openai.requests) == 1
    assert (client.lazy.config.threshold, lenient.lazy.config.threshold) == (0.9, 0.5)
    assert lenient.lazy.fast is client.lazy.fast
    assert isinstance(lenient, lazycontroller.OpenAI)


def test_with_lazy_can_disable_and_pick_a_model(make_client, routes):
    jev = FakeJev(JEV_PICKS)
    client, fake_openai = make_client(jev, OPENAI_TICKET)

    client.with_lazy(enabled=False).chat.completions.parse(
        model="gpt-5", messages=MESSAGES, response_format=Ticket
    )
    client.with_lazy(fast_model="kev:0.8b-q8_0").chat.completions.parse(
        model="gpt-5", messages=MESSAGES, response_format=Ticket
    )

    assert (routes[0].reason, routes[0].detail) == ("disabled", "disabled in LazyConfig")
    assert len(fake_openai.requests) == 1
    assert [r["model"] for r in jev.requests] == ["kev:0.8b-q8_0"]


def test_with_lazy_survives_with_options_and_validates(make_client):
    client, _ = make_client(FakeJev())
    strict = client.with_lazy(threshold=0.99)
    assert strict.with_options(timeout=5).lazy.config.threshold == 0.99
    with pytest.raises(TypeError):
        client.with_lazy(treshold=0.5)
    with pytest.raises(ValueError):
        client.with_lazy(threshold=2)


async def test_async_with_lazy(make_client, routes):
    client, fake_openai = make_client(
        AsyncFakeJev(JEV_PICKS, confidence=0.6), OPENAI_TICKET, is_async=True
    )
    lenient = client.with_lazy(threshold=0.5)
    await lenient.chat.completions.parse(model="gpt-5", messages=MESSAGES, response_format=Ticket)
    assert isinstance(lenient, lazycontroller.AsyncOpenAI)
    assert routes[0].reason == "confident"
    assert fake_openai.requests == []


@pytest.fixture
def auth_server():
    """A local HTTP server that records each request's Authorization header and rejects it."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append(self.headers.get("Authorization"))
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(400)  # not retried by the SDK, unlike 5xx
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", seen
    server.shutdown()


def _parse_through_router(config):
    router = lazycontroller.lazy.Router(config)
    call = lazycontroller.client._chat_parse_call(
        {"model": "gpt-5", "messages": MESSAGES, "response_format": Ticket}
    )
    return router.run(call, lambda: "slow model")


def test_typesafe_key_from_env_never_goes_to_another_server(auth_server, monkeypatch, routes):
    url, seen = auth_server
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-secret")
    assert (
        _parse_through_router(lazycontroller.LazyConfig(fast_base_url=url, on_route=routes.append))
        == "slow model"
    )
    assert seen == ["Bearer local"]
    assert routes[0].reason == "fast_model_error"


def test_explicit_key_goes_to_the_configured_server(auth_server, monkeypatch):
    url, seen = auth_server
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-secret")
    _parse_through_router(lazycontroller.LazyConfig(fast_base_url=url, fast_api_key="provider-key"))
    assert seen == ["Bearer provider-key"]


def test_typesafe_key_from_env_goes_to_typesafe(auth_server, monkeypatch):
    url, seen = auth_server
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-secret")
    monkeypatch.setenv("TYPESAFE_BASE_URL", url)  # stands in for api.typesafe.ai
    _parse_through_router(lazycontroller.LazyConfig())
    assert seen == ["Bearer ts-secret"]


def test_unreadable_request_goes_to_openai(make_client, routes):
    client, fake_openai = make_client(FakeJev(JEV_PICKS), OPENAI_TICKET)
    broken = {
        "type": "json_schema",
        "json_schema": {
            "name": "x",
            "schema": {"type": "object", "properties": {"a": {"$ref": "#/$defs/Missing"}}},
        },
    }
    client.chat.completions.create(model="gpt-5", messages=MESSAGES, response_format=broken)
    assert len(fake_openai.requests) == 1
    assert routes[0].reason == "ineligible"
    assert routes[0].detail.startswith("could not read request: KeyError")


def test_answer_rejected_by_callers_validator_goes_to_openai(make_client, routes):
    from pydantic import field_validator

    class NoTechnical(Ticket):
        @field_validator("team")
        @classmethod
        def not_technical(cls, team):
            if team == "technical":
                raise ValueError("technical is closed today")
            return team

    client, fake_openai = make_client(FakeJev(JEV_PICKS), OPENAI_TICKET)
    completion = client.chat.completions.parse(
        model="gpt-5", messages=MESSAGES, response_format=NoTechnical
    )
    assert completion.choices[0].message.parsed.team == "billing"
    assert len(fake_openai.requests) == 1
    assert routes[0].reason == "fast_model_error"
