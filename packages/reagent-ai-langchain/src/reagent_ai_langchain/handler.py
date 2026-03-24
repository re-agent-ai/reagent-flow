"""LangChain callback handler for reagent-ai."""

from __future__ import annotations

import warnings
from typing import Any
from uuid import UUID

from reagent_ai._context import get_active_session
from reagent_ai.exceptions import ReagentAdapterWarning


class ReagentCallbackHandler:
    """LangChain callback handler that captures tool calls into reagent-ai sessions.

    Implements the LangChain callback interface without importing LangChain,
    making it compatible with any version that supports the callback protocol.
    """

    def __init__(self) -> None:
        self._call_ids: list[str] = []
        self._call_id_index: int = 0

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Handle LLM response completion."""
        session = get_active_session()
        if session is None:
            return

        try:
            generation = response.generations[0][0]
            text = getattr(generation, "text", None)

            tool_calls_data: list[dict[str, Any]] = []
            message = getattr(generation, "message", None)
            if message:
                raw_tool_calls = getattr(message, "tool_calls", None) or []
                for tc in raw_tool_calls:
                    if isinstance(tc, dict):
                        tool_calls_data.append(
                            {
                                "name": tc.get("name", "unknown"),
                                "arguments": tc.get("args", {}),
                                "call_id": tc.get("id", str(run_id)),
                            }
                        )
                    else:
                        tool_calls_data.append(
                            {
                                "name": getattr(tc, "name", "unknown"),
                                "arguments": getattr(tc, "args", {}),
                                "call_id": getattr(tc, "id", str(run_id)),
                            }
                        )

            returned_ids = session.log_llm_call(
                response_text=text,
                tool_calls=tool_calls_data,
            )
            self._call_ids = returned_ids
            self._call_id_index = 0
        except Exception as e:
            warnings.warn(
                ReagentAdapterWarning(f"Failed to capture LangChain LLM response: {e}"),
                stacklevel=2,
            )

    def _next_call_id(self) -> str | None:
        """Return the next tracked call_id, or None if exhausted."""
        if self._call_id_index < len(self._call_ids):
            cid = self._call_ids[self._call_id_index]
            self._call_id_index += 1
            return cid
        return None

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Handle tool execution completion."""
        session = get_active_session()
        if session is None:
            return

        try:
            name = kwargs.get("name", "unknown_tool")
            call_id = self._next_call_id()
            session.log_tool_result(name, result=output, call_id=call_id)
        except Exception as e:
            warnings.warn(
                ReagentAdapterWarning(f"Failed to capture LangChain tool result: {e}"),
                stacklevel=2,
            )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Handle tool execution error."""
        session = get_active_session()
        if session is None:
            return

        try:
            name = kwargs.get("name", "unknown_tool")
            call_id = self._next_call_id()
            session.log_tool_result(name, error=str(error), call_id=call_id)
        except Exception as e:
            warnings.warn(
                ReagentAdapterWarning(f"Failed to capture LangChain tool error: {e}"),
                stacklevel=2,
            )
