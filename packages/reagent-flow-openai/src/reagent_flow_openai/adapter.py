"""OpenAI client adapter for reagent-flow."""

from __future__ import annotations

import json
import warnings
from functools import wraps
from typing import Any

from reagent_flow._context import get_active_session
from reagent_flow.exceptions import ReagentAdapterWarning


def patch(client: Any) -> Any:
    """Wrap an OpenAI client to auto-capture tool calls into the active session."""
    original_create = client.chat.completions.create

    @wraps(original_create)
    def wrapped_create(*args: Any, **kwargs: Any) -> Any:
        response = original_create(*args, **kwargs)
        session = get_active_session()
        if session is None:
            return response

        if kwargs.get("stream"):
            warnings.warn(
                ReagentAdapterWarning(
                    "Streaming responses (stream=True) are not captured by reagent-flow. "
                    "Use stream=False for traced calls."
                ),
                stacklevel=2,
            )
            return response

        try:
            choice = response.choices[0]
            message = choice.message

            tool_calls_data: list[dict[str, Any]] = []
            if message.tool_calls:
                for tc in message.tool_calls:
                    try:
                        arguments = json.loads(tc.function.arguments)
                    except (json.JSONDecodeError, AttributeError):
                        arguments = {}
                    tool_calls_data.append(
                        {
                            "name": tc.function.name,
                            "arguments": arguments,
                            "call_id": tc.id,
                        }
                    )

            session.log_llm_call(
                response_text=message.content,
                tool_calls=tool_calls_data,
                model=getattr(response, "model", None),
                token_usage=_extract_usage(response),
            )
        except Exception as e:
            warnings.warn(
                ReagentAdapterWarning(f"Failed to capture OpenAI response: {e}"),
                stacklevel=2,
            )

        return response

    client.chat.completions.create = wrapped_create
    return client


def _extract_usage(response: Any) -> dict[str, Any] | None:
    """Extract token usage from OpenAI response."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0),
        "completion_tokens": getattr(usage, "completion_tokens", 0),
    }
