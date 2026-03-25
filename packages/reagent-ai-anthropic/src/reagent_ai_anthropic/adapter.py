"""Anthropic client adapter for reagent-ai."""

from __future__ import annotations

import warnings
from functools import wraps
from typing import Any

from reagent_ai._context import get_active_session
from reagent_ai.exceptions import ReagentAdapterWarning


def patch(client: Any) -> Any:
    """Wrap an Anthropic client to auto-capture tool calls into the active session.

    Patches ``client.messages.create`` to log tool_use blocks as tool calls.
    Streaming is detected and skipped with a warning.
    """
    original_create = client.messages.create

    @wraps(original_create)
    def wrapped_create(*args: Any, **kwargs: Any) -> Any:
        response = original_create(*args, **kwargs)
        session = get_active_session()
        if session is None:
            return response

        if kwargs.get("stream"):
            warnings.warn(
                ReagentAdapterWarning(
                    "Streaming responses (stream=True) are not captured by reagent-ai. "
                    "Use stream=False for traced calls."
                ),
                stacklevel=2,
            )
            return response

        try:
            tool_calls_data: list[dict[str, Any]] = []
            response_text: str | None = None

            for block in response.content:
                if block.type == "tool_use":
                    arguments = block.input if isinstance(block.input, dict) else {}
                    tool_calls_data.append(
                        {
                            "name": block.name,
                            "arguments": arguments,
                            "call_id": block.id,
                        }
                    )
                elif block.type == "text":
                    response_text = block.text

            session.log_llm_call(
                response_text=response_text,
                tool_calls=tool_calls_data,
                model=getattr(response, "model", None),
                token_usage=_extract_usage(response),
            )
        except Exception as e:
            warnings.warn(
                ReagentAdapterWarning(f"Failed to capture Anthropic response: {e}"),
                stacklevel=2,
            )

        return response

    client.messages.create = wrapped_create
    return client


def _extract_usage(response: Any) -> dict[str, Any] | None:
    """Extract token usage from Anthropic response."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return {
        "input_tokens": getattr(usage, "input_tokens", 0),
        "output_tokens": getattr(usage, "output_tokens", 0),
    }
