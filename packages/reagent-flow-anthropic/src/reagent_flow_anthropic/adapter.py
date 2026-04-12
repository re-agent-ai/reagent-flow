"""Anthropic client adapter for reagent-flow."""

from __future__ import annotations

import json
import warnings
from functools import wraps
from typing import Any

from reagent_flow._context import get_active_session
from reagent_flow.exceptions import ReagentAdapterWarning


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
                ReagentAdapterWarning(f"Failed to capture Anthropic tool results: {e}"),
                stacklevel=2,
            )

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


def _log_prior_tool_results(session: Any, messages: Any) -> None:
    """Log tool results found inside user messages of the outgoing call.

    Anthropic threads tool execution results back to the model as user
    messages whose ``content`` is a list containing
    ``{"type": "tool_result", "tool_use_id": ..., "content": ...}`` blocks.
    We look up any such blocks whose ``tool_use_id`` matches a still-pending
    call in the recorder and log them via ``session.log_tool_result``. They
    attach to the turn that originally issued the call, which is where
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
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_result":
                continue
            call_id = block.get("tool_use_id")
            if not call_id or call_id not in pending:
                continue
            name = pending[call_id].name
            session.log_tool_result(
                name,
                call_id=call_id,
                result=_parse_tool_result_content(block.get("content")),
            )


def _parse_tool_result_content(content: Any) -> Any:
    """Parse an Anthropic ``tool_result`` content field into a structured value.

    Anthropic allows ``content`` to be either a plain string or a list of
    content blocks (``{"type": "text", "text": ...}``). When the extracted
    text looks like JSON, we decode it so schema assertions can validate
    real dict/list shapes rather than opaque strings.
    """
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
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
