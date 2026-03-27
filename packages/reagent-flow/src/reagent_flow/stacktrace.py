"""Agent Stack Trace formatter with probable cause detection."""

from __future__ import annotations

import json
from typing import Any

from reagent_flow.models import Trace


def _fmt_args(arguments: dict[str, Any]) -> str:
    """Format tool arguments for display."""
    parts: list[str] = []
    for k, v in arguments.items():
        parts.append(f"{k}={json.dumps(v)}")
    return ", ".join(parts)


def _fmt_result(result: Any) -> str:
    """Format a tool result for display."""
    return json.dumps(result, default=str)


def _find_probable_cause(trace: Trace, expected_tool: str | None) -> str | None:
    """Heuristic probable cause detection."""
    if not expected_tool:
        return None

    last_tool_name: str | None = None
    last_tool_result: Any = None
    last_errored: tuple[int, str] | None = None

    for turn in trace.turns:
        for tc in turn.llm_call.tool_calls:
            for tr in turn.tool_results:
                if tr.call_id == tc.call_id:
                    last_tool_name = tc.name
                    last_tool_result = tr.result if not tr.error else f"ERROR: {tr.error}"
        for tr in turn.tool_results:
            if tr.error:
                last_errored = (turn.index, tr.error)

    lines: list[str] = []
    if last_errored:
        idx, err = last_errored
        lines.append(f"  Turn {idx} had a tool error: {err}")
        lines.append("  LLM may have stopped after the error")
    elif last_tool_name and last_tool_result is not None:
        lines.append("  Last tool output before termination:")
        lines.append(f"    {last_tool_name} -> {_fmt_result(last_tool_result)}")

    if lines:
        return "\n".join(lines)
    return None


def format_stack_trace(
    trace: Trace,
    *,
    assertion_msg: str,
    expected_tool: str | None = None,
    color: bool = True,
) -> str:
    """Format an Agent Stack Trace for display."""
    sep = "=" * 54
    lines: list[str] = []

    lines.append(sep)
    lines.append(f"AGENT STACK TRACE -- {trace.name}")
    lines.append(sep)
    lines.append("")

    for turn in trace.turns:
        has_error = any(tr.error for tr in turn.tool_results)
        status_char = "\u2717" if has_error else "\u2713"

        dur = f" [{turn.duration_ms:.0f}ms]" if turn.duration_ms else ""
        lines.append(f"{status_char} Turn {turn.index}{dur}")

        if turn.llm_call.tool_calls:
            for tc in turn.llm_call.tool_calls:
                args_str = _fmt_args(tc.arguments)
                lines.append(f"  LLM -> {tc.name}({args_str})")
        elif turn.llm_call.response_text:
            text = turn.llm_call.response_text
            if len(text) > 80:
                text = text[:77] + "..."
            lines.append(f'  LLM -> [TEXT] "{text}"')
            lines.append("  No tool calls made.")

        for tr in turn.tool_results:
            if tr.error:
                lines.append(f"  Error: {tr.error}")
            else:
                lines.append(f"  Result: {_fmt_result(tr.result)}")

        lines.append("")

    lines.append(sep)
    lines.append(f"ASSERTION FAILED: {assertion_msg}")

    cause = _find_probable_cause(trace, expected_tool)
    if cause:
        lines.append("")
        lines.append("PROBABLE CAUSE:")
        lines.append(cause)

    lines.append(sep)

    output = "\n".join(lines)

    if not color:
        return output

    output = output.replace("\u2713", "\033[32m\u2713\033[0m")
    output = output.replace("\u2717", "\033[31m\u2717\033[0m")
    output = output.replace("ASSERTION FAILED:", "\033[31mASSERTION FAILED:\033[0m")
    output = output.replace("PROBABLE CAUSE:", "\033[33mPROBABLE CAUSE:\033[0m")

    return output
