"""Build OpenAI response objects carrying the fast model's answer, indistinguishable to callers."""

import json
import time
import uuid
from collections.abc import Mapping
from typing import Any

# The SDK's own lenient constructor: a field added in a later SDK version must not break us.
from openai._models import construct_type_unchecked
from openai.types.chat import ChatCompletion
from openai.types.responses import Response
from typesafe_sdk import SystemOneResponse


def chat_completion(output: Mapping[str, Any], answer: SystemOneResponse) -> ChatCompletion:
    usage = _usage(answer)
    return construct_type_unchecked(
        type_=ChatCompletion,
        value={
            "id": f"chatcmpl-lazy-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": answer.model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "logprobs": None,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(output),
                        "refusal": None,
                    },
                }
            ],
            "usage": {
                "prompt_tokens": usage[0],
                "completion_tokens": usage[1],
                "total_tokens": usage[0] + usage[1],
            },
        },
    )


def response(
    output: Mapping[str, Any], answer: SystemOneResponse, kwargs: Mapping[str, Any]
) -> Response:
    usage = _usage(answer)
    text = kwargs.get("text") or {}
    return construct_type_unchecked(
        type_=Response,
        value={
            "id": f"resp_lazy_{uuid.uuid4().hex}",
            "object": "response",
            "created_at": time.time(),
            "status": "completed",
            "model": answer.model,
            "output": [
                {
                    "type": "message",
                    "id": f"msg_lazy_{uuid.uuid4().hex}",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps(output),
                            "annotations": [],
                        }
                    ],
                }
            ],
            "instructions": kwargs.get("instructions"),
            "metadata": kwargs.get("metadata"),
            "text": text,
            "tools": [],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "temperature": kwargs.get("temperature"),
            "top_p": kwargs.get("top_p"),
            "usage": {
                "input_tokens": usage[0],
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens": usage[1],
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": usage[0] + usage[1],
            },
        },
    )


def _usage(answer: SystemOneResponse) -> tuple[int, int]:
    return answer.usage.input_tokens or 0, answer.usage.output_tokens or 0
