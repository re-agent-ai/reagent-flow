"""LangChain callback handler for reagent-flow."""

from __future__ import annotations

import warnings
from typing import Any
from uuid import UUID

from reagent_flow._context import get_active_session
from reagent_flow.exceptions import ReagentAdapterWarning


class ReagentCallbackHandler:
    """LangChain callback handler that captures tool calls into reagent-flow sessions.

    Implements the LangChain callback interface without importing LangChain,
    making it compatible with any version that supports the callback protocol.
    """

    def __init__(self) -> None:
        self._call_id_by_name: dict[str, list[str]] = {}
        self._call_id_by_run_id: dict[str, str] = {}

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
            # Build a name -> [call_ids] map for correlation by tool name
            self._call_id_by_name = {}
            for tc_data, cid in zip(tool_calls_data, returned_ids):
                name = tc_data["name"]
                self._call_id_by_name.setdefault(name, []).append(cid)
            self._call_id_by_run_id = {}
        except Exception as e:
            warnings.warn(
                ReagentAdapterWarning(f"Failed to capture LangChain LLM response: {e}"),
                stacklevel=2,
            )

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Map a tool run_id to its call_id at start time.

        LangChain fires on_tool_start before the tool executes, giving us the
        same run_id that will arrive in on_tool_end/on_tool_error. By consuming
        the call_id from the per-name queue here, we guarantee correct
        correlation even when same-name tools finish out of order.
        """
        name = serialized.get("name") or kwargs.get("name", "unknown_tool")
        ids = self._call_id_by_name.get(name, [])
        if ids:
            self._call_id_by_run_id[str(run_id)] = ids.pop(0)

    def _pop_call_id(self, name: str, run_id: UUID) -> str | None:
        """Return the call_id for a tool completion, correlating by run_id."""
        run_key = str(run_id)
        if run_key in self._call_id_by_run_id:
            return self._call_id_by_run_id.pop(run_key)
        # Fallback: consume next call_id for this tool name (covers cases
        # where on_tool_start was not fired, e.g. older LangChain versions)
        ids = self._call_id_by_name.get(name, [])
        if ids:
            return ids.pop(0)
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
            call_id = self._pop_call_id(name, run_id)
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
            call_id = self._pop_call_id(name, run_id)
            session.log_tool_result(name, error=str(error), call_id=call_id)
        except Exception as e:
            warnings.warn(
                ReagentAdapterWarning(f"Failed to capture LangChain tool error: {e}"),
                stacklevel=2,
            )
