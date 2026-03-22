"""Recorder: builds Turn objects from log events."""

from __future__ import annotations

import time
import uuid
from typing import Any

from ttrace_ai.exceptions import AmbiguousToolCallError
from ttrace_ai.models import LLMCall, Message, ToolCall, ToolResult, Turn


class Recorder:
    """Builds Turn objects from log_llm_call / log_tool_result events."""

    def __init__(self) -> None:
        self.turns: list[Turn] = []
        self._current_turn: Turn | None = None
        self._pending_calls: dict[str, ToolCall] = {}

    def log_llm_call(
        self,
        *,
        messages: list[dict[str, Any]] | None = None,
        response_text: str | None = None,
        tool_calls: list[dict[str, Any]],
        model: str | None = None,
        token_usage: dict[str, Any] | None = None,
    ) -> list[str]:
        """Record an LLM call and return generated call_ids."""
        now = time.time()
        parsed_messages: list[Message] | None = None
        if messages:
            parsed_messages = [Message(role=m["role"], content=m["content"]) for m in messages]

        parsed_tool_calls: list[ToolCall] = []
        call_ids: list[str] = []
        for tc in tool_calls:
            call_id = tc.get("call_id", str(uuid.uuid4()))
            call_ids.append(call_id)
            tool_call = ToolCall(
                name=tc["name"],
                arguments=tc.get("arguments", {}),
                call_id=call_id,
                timestamp=now,
            )
            parsed_tool_calls.append(tool_call)
            self._pending_calls[call_id] = tool_call

        llm_call = LLMCall(
            messages=parsed_messages,
            response_text=response_text,
            tool_calls=parsed_tool_calls,
            model=model,
            token_usage=token_usage,
            timestamp=now,
        )

        turn = Turn(index=len(self.turns), llm_call=llm_call, tool_results=[])
        self._current_turn = turn
        self.turns.append(turn)
        return call_ids

    def log_tool_result(
        self,
        name: str,
        *,
        call_id: str | None = None,
        result: Any = None,
        error: str | None = None,
        duration_ms: float = 0,
    ) -> None:
        """Record a tool result for the current turn."""
        if self._current_turn is None:
            return

        if call_id is None:
            matching = [
                tc
                for tc in self._current_turn.llm_call.tool_calls
                if tc.name == name and tc.call_id in self._pending_calls
            ]
            if len(matching) > 1:
                raise AmbiguousToolCallError(
                    f"Multiple pending calls for '{name}'. Provide call_id to disambiguate."
                )
            if matching:
                call_id = matching[0].call_id
            else:
                call_id = str(uuid.uuid4())

        self._pending_calls.pop(call_id, None)

        tool_result = ToolResult(
            call_id=call_id,
            result=result,
            error=error,
            duration_ms=duration_ms,
        )
        self._current_turn.tool_results.append(tool_result)
