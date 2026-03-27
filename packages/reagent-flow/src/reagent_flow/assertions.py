"""Assertion implementations for reagent-flow sessions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from reagent_flow.stacktrace import format_stack_trace

if TYPE_CHECKING:
    from reagent_flow.models import Trace


def _assertion_error(trace: Trace, msg: str, expected_tool: str | None = None) -> AssertionError:
    """Create an AssertionError with Agent Stack Trace attached."""
    stack = format_stack_trace(trace, assertion_msg=msg, expected_tool=expected_tool, color=True)
    return AssertionError(f"\n{stack}")


def _all_tool_names(trace: Trace) -> set[str]:
    """Return set of all tool names called in the trace."""
    names: set[str] = set()
    for turn in trace.turns:
        for tc in turn.llm_call.tool_calls:
            names.add(tc.name)
    return names


def _tool_call_count(trace: Trace) -> int:
    """Return total number of tool calls."""
    return sum(len(turn.llm_call.tool_calls) for turn in trace.turns)


def assert_called(trace: Trace, tool_name: str) -> None:
    """Assert that a tool was called at least once."""
    if tool_name not in _all_tool_names(trace):
        total = _tool_call_count(trace)
        raise _assertion_error(
            trace,
            f'"{tool_name}" was never called ({len(trace.turns)} turns, {total} tool calls)',
            expected_tool=tool_name,
        )


def assert_never_called(trace: Trace, tool_name: str) -> None:
    """Assert that a tool was never called."""
    if tool_name not in _all_tool_names(trace):
        return
    indices = [
        turn.index
        for turn in trace.turns
        for tc in turn.llm_call.tool_calls
        if tc.name == tool_name
    ]
    raise _assertion_error(
        trace,
        f'"{tool_name}" was called in turn(s) {indices}',
    )


def assert_called_before(trace: Trace, first: str, second: str) -> None:
    """Assert that first tool was called before second tool."""
    first_idx: int | None = None
    second_idx: int | None = None
    for turn in trace.turns:
        for tc in turn.llm_call.tool_calls:
            if tc.name == first and first_idx is None:
                first_idx = turn.index
            if tc.name == second and second_idx is None:
                second_idx = turn.index
    if first_idx is None:
        raise _assertion_error(trace, f'"{first}" was never called')
    if second_idx is None:
        raise _assertion_error(trace, f'"{second}" was never called')
    if first_idx >= second_idx:
        raise _assertion_error(
            trace,
            f'"{first}" (turn {first_idx}) was not called before "{second}" (turn {second_idx})',
        )


def assert_tool_succeeded(trace: Trace, tool_name: str) -> None:
    """Assert that a tool was called and all its executions succeeded."""
    found = False
    for turn in trace.turns:
        for tc in turn.llm_call.tool_calls:
            if tc.name == tool_name:
                found = True
                matching_results = [tr for tr in turn.tool_results if tr.call_id == tc.call_id]
                if not matching_results:
                    raise _assertion_error(
                        trace,
                        f'"{tool_name}" in turn {turn.index} has no recorded result',
                        expected_tool=tool_name,
                    )
                for tr in matching_results:
                    if tr.error is not None:
                        raise _assertion_error(
                            trace,
                            f'"{tool_name}" failed in turn {turn.index}: {tr.error}',
                            expected_tool=tool_name,
                        )
    if not found:
        raise _assertion_error(
            trace,
            f'"{tool_name}" was never called ({len(trace.turns)} turns, '
            f"{_tool_call_count(trace)} tool calls)",
            expected_tool=tool_name,
        )


def assert_max_turns(trace: Trace, n: int) -> None:
    """Assert that the trace has at most n turns."""
    if len(trace.turns) > n:
        raise _assertion_error(trace, f"Expected at most {n} turns, got {len(trace.turns)}")


def assert_total_duration_under(trace: Trace, *, ms: float) -> None:
    """Assert that total trace duration is under ms milliseconds.

    Works inside active sessions by using the current wall clock time
    when ``ended_at`` has not been set yet.
    """
    import time

    end = trace.ended_at if trace.ended_at is not None else time.time()
    actual_ms = (end - trace.started_at) * 1000
    if actual_ms > ms:
        raise _assertion_error(
            trace,
            f"Total duration {actual_ms:.0f}ms exceeds limit of {ms:.0f}ms",
        )
