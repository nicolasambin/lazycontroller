"""Turn a structured-output JSON schema into decision questions, and the answers back into JSON.

A schema is eligible only when every value it asks for is a closed decision:

- ``enum`` (from ``Literal``, ``Enum``, ...) becomes a Choice;
- ``boolean`` becomes a Noul;
- ``integer`` with a small ``minimum``..``maximum`` range becomes a Score, one level per value;
- ``const`` is filled in without asking;
- objects are walked field by field, and ``X | None`` of an enum or boolean becomes a Choice
  with a ``null`` option.

Anything else (free strings, arrays, unbounded numbers, unions) needs a generative model, so the
whole request is left to the slow model.
"""

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from typesafe_sdk import (
    Answer,
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
)

MAX_SCORE_LEVELS = 11
"""Integer ranges wider than this are treated as open numbers, not a rubric."""

_MAX_DEPTH = 16

# Keywords whose meaning we honour. Any other constraint (multipleOf, pattern, format, ...)
# could reject a value the fast model picks, so it makes the request ineligible rather than ignored.
_ANNOTATIONS = {"title", "description", "default", "examples", "$comment", "deprecated"}
_KEYWORDS = _ANNOTATIONS | {
    "type",
    "enum",
    "const",
    "properties",
    "required",
    "additionalProperties",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "$defs",
    "definitions",
}

type Path = tuple[str, ...]
type Decoder = Callable[[Any], tuple[Any, float]]


class Ineligible(Exception):
    """The fast model can't answer this request; the message says why."""


@dataclass(frozen=True)
class Field:
    path: Path
    question: Choice | Noul | Score
    decode: Decoder
    """Maps the decision answer to (JSON value, confidence in 0..1)."""


@dataclass(frozen=True)
class Plan:
    """The decision questions for one schema, plus the values that need no question."""

    fields: tuple[Field, ...]
    constants: tuple[tuple[Path, Any], ...]

    @property
    def questions(self) -> dict[str, Choice | Noul | Score]:
        return {_question_id(i, f.path): f.question for i, f in enumerate(self.fields)}

    def assemble(self, answers: Mapping[str, Answer]) -> tuple[dict[str, Any], dict[str, float]]:
        """Build the output object; also return each field's confidence keyed by dotted path."""
        output: dict[str, Any] = {}
        confidences: dict[str, float] = {}
        for path, value in self.constants:
            _put(output, path, value)
        for i, f in enumerate(self.fields):
            value, confidence = f.decode(answers[_question_id(i, f.path)])
            _put(output, f.path, value)
            confidences[".".join(f.path)] = confidence
        return output, confidences


def compile_schema(schema: Mapping[str, Any], *, title: str | None = None) -> Plan:
    """Raise Ineligible unless every value in ``schema`` maps to a decision question."""
    fields: list[Field] = []
    constants: list[tuple[Path, Any]] = []
    _walk(schema, schema, (), title or schema.get("title"), fields, constants, 0)
    if not fields:
        raise Ineligible("schema has no fields to decide")
    return Plan(tuple(fields), tuple(constants))


def _walk(
    node: Mapping[str, Any],
    root: Mapping[str, Any],
    path: Path,
    title: str | None,
    fields: list[Field],
    constants: list[tuple[Path, Any]],
    depth: int,
) -> None:
    if depth > _MAX_DEPTH:
        raise Ineligible("schema is nested too deeply")
    node = _resolve(node, root)
    where = ".".join(path) or "<root>"

    if "const" in node:
        if not path:
            raise Ineligible("root schema is a constant")
        constants.append((path, node["const"]))
        return

    if _type(node) == "object" or "properties" in node:
        if not path and _type(node) != "object":
            raise Ineligible("root schema is not an object")
        properties = node.get("properties") or {}
        if not properties:
            raise Ineligible(f"{where}: object without properties")
        if node.get("patternProperties") or isinstance(node.get("additionalProperties"), dict):
            raise Ineligible(f"{where}: open-ended object")
        _check_keywords(node, where)
        for name, child in properties.items():
            _walk(child, root, (*path, name), title, fields, constants, depth + 1)
        return

    if not path:
        raise Ineligible("root schema is not an object")
    fields.append(_leaf(node, root, path, title))


def _leaf(node: Mapping[str, Any], root: Mapping[str, Any], path: Path, title: str | None) -> Field:
    where = ".".join(path)
    instructions = _instructions(node, path, title)

    nullable, node = _split_nullable(node, root)
    field = _leaf_field(node, path, instructions, nullable)
    _check_keywords(node, where)
    return field


def _leaf_field(node: Mapping[str, Any], path: Path, instructions: str, nullable: bool) -> Field:
    where = ".".join(path)
    if "enum" in node:
        return _choice(path, instructions, list(node["enum"]), nullable)
    match _type(node):
        case "boolean" if nullable:
            return _choice(path, instructions, [True, False], nullable=True)
        case "boolean":
            return Field(path, Noul(instructions=instructions), _decode_noul)
        case "integer" if not nullable:
            return _score(path, instructions, node)
        case "string":
            raise Ineligible(f"{where}: free-text string")
        case "array":
            raise Ineligible(f"{where}: array")
        case "number" | "integer":
            raise Ineligible(f"{where}: open-ended number")
        case _:
            raise Ineligible(f"{where}: unsupported schema {json.dumps(node)[:80]}")


def _choice(path: Path, instructions: str, values: list[Any], nullable: bool) -> Field:
    if nullable and None not in values:
        values = [*values, None]
    if not values:
        raise Ineligible(f"{'.'.join(path)}: empty enum")
    labels = {json.dumps(v) if not isinstance(v, str) else v: v for v in values}
    if len(labels) != len(values):
        raise Ineligible(f"{'.'.join(path)}: enum values collide as labels")
    criteria = {
        label: ("Not stated or not applicable" if v is None else None)
        for label, v in labels.items()
    }

    def decode(answer: Any) -> tuple[Any, float]:
        if not isinstance(answer, ChoiceAnswer) or answer.choice not in labels:
            raise TypeError(f"expected a choice among {list(labels)}, got {answer!r}")
        return labels[answer.choice], answer.confidence

    return Field(path, Choice(instructions=instructions, criteria=criteria), decode)


def _decode_noul(answer: Any) -> tuple[Any, float]:
    if not isinstance(answer, NoulAnswer):
        raise TypeError(f"expected a noul, got {answer!r}")
    # Nouls carry no confidence; |2p - 1| is the one TypeSafe suggests, on the Choice scale.
    return answer.noul >= 0.5, abs(2 * answer.noul - 1)


def _score(path: Path, instructions: str, node: Mapping[str, Any]) -> Field:
    where = ".".join(path)
    low, high = node.get("minimum"), node.get("maximum")
    if "exclusiveMinimum" in node:
        low = node["exclusiveMinimum"] + 1
    if "exclusiveMaximum" in node:
        high = node["exclusiveMaximum"] - 1
    if not isinstance(low, int) or not isinstance(high, int):
        raise Ineligible(f"{where}: integer without minimum and maximum")
    if not 2 <= high - low + 1 <= MAX_SCORE_LEVELS:
        raise Ineligible(f"{where}: integer range {low}..{high} is not a small rubric")
    levels = [f"{path[-1]} = {value}" for value in range(low, high + 1)]

    def decode(answer: Any) -> tuple[Any, float]:
        if not isinstance(answer, ScoreAnswer):
            raise TypeError(f"expected a score, got {answer!r}")
        # The most likely level, not the expected score: the schema wants one of its integers.
        level = max(answer.probabilities, key=answer.probabilities.__getitem__)
        return low + level, answer.confidence

    return Field(path, Score(instructions=instructions, criteria=levels), decode)


def _split_nullable(
    node: Mapping[str, Any], root: Mapping[str, Any]
) -> tuple[bool, Mapping[str, Any]]:
    """Unwrap ``X | None`` written as anyOf/oneOf or as a ``type`` list."""
    for key in ("anyOf", "oneOf"):
        if key in node:
            options = [_resolve(o, root) for o in node[key]]
            rest = [o for o in options if _type(o) != "null"]
            if len(rest) != 1:
                raise Ineligible("union of several types")
            merged = {**{k: v for k, v in node.items() if k != key}, **rest[0]}
            return len(rest) < len(options), merged
    types = node.get("type")
    if isinstance(types, list):
        rest = [t for t in types if t != "null"]
        if len(rest) != 1:
            raise Ineligible("union of several types")
        return len(rest) < len(types), {**node, "type": rest[0]}
    return False, node


def _check_keywords(node: Mapping[str, Any], where: str) -> None:
    unknown = sorted(set(node) - _KEYWORDS)
    if unknown:
        raise Ineligible(f"{where}: unsupported keyword {', '.join(unknown)}")
    if node.get("additionalProperties") not in (None, False, True):
        raise Ineligible(f"{where}: open-ended object")


def _resolve(node: Mapping[str, Any], root: Mapping[str, Any]) -> Mapping[str, Any]:
    seen = 0
    while "$ref" in node:
        ref = node["$ref"]
        if not ref.startswith("#/"):
            raise Ineligible(f"external $ref {ref}")
        target: Any = root
        for part in ref[2:].split("/"):
            target = target[part.replace("~1", "/").replace("~0", "~")]
        # Sibling keywords (e.g. a field description next to an Enum $ref) win over the target.
        node = {**target, **{k: v for k, v in node.items() if k != "$ref"}}
        seen += 1
        if seen > _MAX_DEPTH:
            raise Ineligible("recursive $ref")
    if "allOf" in node and len(node["allOf"]) == 1:
        inner = _resolve(node["allOf"][0], root)
        node = {**inner, **{k: v for k, v in node.items() if k != "allOf"}}
    if "allOf" in node:
        raise Ineligible("allOf with several schemas")
    return node


def _type(node: Mapping[str, Any]) -> Any:
    return node.get("type")


def _instructions(node: Mapping[str, Any], path: Path, title: str | None) -> str:
    output = title or "structured output"
    subject = f"the `{'.'.join(path)}` field of the {output} the conversation asks for"
    description = node.get("description") or _humanize(path[-1])
    return f"Decide {subject}: {description}"


def _humanize(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|_+", " ", name).strip().lower()


def _question_id(index: int, path: Path) -> str:
    return f"f{index}_" + re.sub(r"[^A-Za-z0-9_]", "_", "__".join(path))[:60]


def _put(output: dict[str, Any], path: Path, value: Any) -> None:
    for key in path[:-1]:
        output = output.setdefault(key, {})
    output[path[-1]] = value
