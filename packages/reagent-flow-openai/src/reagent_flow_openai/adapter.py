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

        # Log any tool results that the user is sending back in this call's
        # messages before we record the new LLM turn, so they attach to the
        # turn that originally requested them.
        try:
            _log_prior_tool_results(session, kwargs.get("messages"))
        except Exception as e:
            warnings.warn(
                ReagentAdapterWarning(f"Failed to capture OpenAI tool results: {e}"),
                stacklevel=2,
            )

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


def _log_prior_tool_results(session: Any, messages: Any) -> None:
    """Log tool results found in the outgoing messages list.

    OpenAI threads tool execution results back to the model as
    ``{"role": "tool", "tool_call_id": ..., "content": ...}`` entries in the
    next ``chat.completions.create`` call. We look up any such entries whose
    ``tool_call_id`` matches a still-pending call in the recorder and log
    them via ``session.log_tool_result``. They attach to the turn that
    originally issued the call (the recorder's current turn), which is where
    ``assert_tool_output_matches`` expects to find them.
    """
    if not messages:
        return
    pending = session._recorder._pending_calls
    if not pending:
        return
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "tool":
            continue
        call_id = msg.get("tool_call_id")
        if not call_id or call_id not in pending:
            continue
        name = pending[call_id].name
        session.log_tool_result(
            name,
            call_id=call_id,
            result=_parse_tool_content(msg.get("content")),
        )


def _parse_tool_content(content: Any) -> Any:
    """Parse an OpenAI tool-message ``content`` field into a structured value.

    OpenAI tool messages may carry either a plain string or a list of
    content blocks. When the extracted text looks like JSON, we decode it so
    schema assertions can validate real dict/list shapes rather than opaque
    strings.
    """
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            content = "".join(parts)
    if isinstance(content, str):
        stripped = content.strip()
        if stripped.startswith(("{", "[")):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return content
    return content
