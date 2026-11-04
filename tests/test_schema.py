from enum import Enum
from typing import Literal

import pytest
from openai.lib._pydantic import to_strict_json_schema
from pydantic import BaseModel, Field
from typesafe_sdk import Choice, Noul, Score

from lazycontroller.schema import Ineligible, compile_schema


class Team(Enum):
    BILLING = "billing"
    TECHNICAL = "technical"


class Address(BaseModel):
    country: Literal["IT", "FR", "DE"]


class Ticket(BaseModel):
    team: Team = Field(description="Which team should handle this")
    priority: Literal["low", "high"]
    is_urgent: bool
    severity: int = Field(ge=1, le=5)
    language: Literal["it", "en"] | None
    kind: Literal["ticket"]
    address: Address


def plan_for(model):
    return compile_schema(to_strict_json_schema(model), title=model.__name__)


def test_pydantic_model_maps_to_questions():
    plan = plan_for(Ticket)
    by_path = {".".join(f.path): f.question for f in plan.fields}
    assert set(by_path) == {
        "team",
        "priority",
        "is_urgent",
        "severity",
        "language",
        "address.country",
    }
    assert isinstance(by_path["team"], Choice)
    assert list(by_path["team"].criteria) == ["billing", "technical"]
    assert "Which team should handle this" in by_path["team"].instructions
    assert isinstance(by_path["is_urgent"], Noul)
    assert isinstance(by_path["severity"], Score)
    assert len(by_path["severity"].criteria) == 5
    assert list(by_path["language"].criteria) == ["it", "en", "null"]
    assert list(by_path["address.country"].criteria) == ["IT", "FR", "DE"]
    assert plan.constants == ((("kind",), "ticket"),)
    assert all("Ticket" in q.instructions for q in by_path.values())


def test_question_ids_are_unique_and_safe():
    ids = list(plan_for(Ticket).questions)
    assert len(set(ids)) == len(ids)
    assert all(qid.replace("_", "").isalnum() for qid in ids)


class Summary(BaseModel):
    label: Literal["a", "b"]
    summary: str


class Tags(BaseModel):
    tags: list[Literal["a", "b"]]


class Amount(BaseModel):
    amount: float


class Wide(BaseModel):
    rating: int = Field(ge=0, le=100)


class Unbounded(BaseModel):
    count: int


class Mixed(BaseModel):
    value: int | bool


@pytest.mark.parametrize(
    ("model", "reason"),
    [
        (Summary, "summary: free-text string"),
        (Tags, "tags: array"),
        (Amount, "amount: open-ended number"),
        (Wide, "not a small rubric"),
        (Unbounded, "count: integer without minimum and maximum"),
        (Mixed, "union of several types"),
    ],
)
def test_generative_fields_make_the_schema_ineligible(model, reason):
    with pytest.raises(Ineligible, match=reason):
        plan_for(model)


def test_raw_json_schema_with_type_lists_and_defs():
    schema = {
        "type": "object",
        "$defs": {"Mood": {"type": "string", "enum": ["calm", "angry"]}},
        "properties": {
            "mood": {"$ref": "#/$defs/Mood", "description": "How the customer feels"},
            "refund": {"type": ["boolean", "null"]},
        },
        "required": ["mood", "refund"],
        "additionalProperties": False,
    }
    plan = compile_schema(schema)
    mood, refund = plan.fields
    assert "How the customer feels" in mood.question.instructions
    assert list(mood.question.criteria) == ["calm", "angry"]
    assert list(refund.question.criteria) == ["true", "false", "null"]


def test_schema_without_decisions_is_ineligible():
    with pytest.raises(Ineligible):
        compile_schema({"type": "object", "properties": {"kind": {"const": "x"}}})


class Even(BaseModel):
    size: int = Field(ge=1, le=5, multiple_of=2)


def test_constraints_we_dont_honour_make_the_schema_ineligible():
    with pytest.raises(Ineligible, match="size: unsupported keyword multipleOf"):
        plan_for(Even)
    schema = {
        "type": "object",
        "properties": {"code": {"type": "string", "enum": ["a1", "b2"], "pattern": "^a"}},
    }
    with pytest.raises(Ineligible, match="code: unsupported keyword pattern"):
        compile_schema(schema)
