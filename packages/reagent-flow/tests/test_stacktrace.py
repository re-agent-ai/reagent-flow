"""Tests for Agent Stack Trace formatter."""

from reagent_flow.models import LLMCall, ToolCall, ToolResult, Trace, Turn
from reagent_flow.stacktrace import format_stack_trace


def _make_3turn_trace() -> Trace:
    t0 = Turn(
        index=0,
        llm_call=LLMCall(
            messages=None,
            response_text=None,
            tool_calls=[
                ToolCall(name="lookup_order", arguments={"id": "123"}, call_id="c0", timestamp=1.0),
            ],
            timestamp=1.0,
        ),
        tool_results=[ToolResult(call_id="c0", result={"status": "shipped"}, duration_ms=120)],
        duration_ms=120,
    )
    t1 = Turn(
        index=1,
        llm_call=LLMCall(
            messages=None,
            response_text=None,
            tool_calls=[
                ToolCall(name="check_policy", arguments={"id": "123"}, call_id="c1", timestamp=2.0),
            ],
            timestamp=2.0,
        ),
        tool_results=[ToolResult(call_id="c1", result={"eligible": False}, duration_ms=85)],
        duration_ms=85,
    )
    t2 = Turn(
        index=2,
        llm_call=LLMCall(
            messages=None,
            response_text="Cannot process refund.",
            tool_calls=[],
            timestamp=3.0,
        ),
        tool_results=[],
        duration_ms=40,
    )
    return Trace(
        trace_id="t1",
        name="test_refund",
        turns=[t0, t1, t2],
        started_at=1.0,
        ended_at=4.0,
    )


def test_format_stack_trace_contains_turns() -> None:
    trace = _make_3turn_trace()
    output = format_stack_trace(trace, assertion_msg='assert_called("process_refund") FAILED')
    assert "Turn 0" in output
    assert "Turn 1" in output
    assert "Turn 2" in output
    assert "lookup_order" in output
    assert "check_policy" in output


def test_format_stack_trace_contains_assertion() -> None:
    trace = _make_3turn_trace()
    output = format_stack_trace(trace, assertion_msg='"process_refund" was never called')
    assert "process_refund" in output
    assert "never called" in output


def test_format_stack_trace_shows_probable_cause() -> None:
    trace = _make_3turn_trace()
    output = format_stack_trace(
        trace,
        assertion_msg='"process_refund" was never called',
        expected_tool="process_refund",
    )
    assert "PROBABLE CAUSE" in output or "probable cause" in output.lower()


def test_format_stack_trace_tool_error() -> None:
    t0 = Turn(
        index=0,
        llm_call=LLMCall(
            messages=None,
            response_text=None,
            tool_calls=[ToolCall(name="lookup", arguments={}, call_id="c0", timestamp=1.0)],
            timestamp=1.0,
        ),
        tool_results=[ToolResult(call_id="c0", result=None, error="timeout", duration_ms=5000)],
        duration_ms=5000,
    )
    t1 = Turn(
        index=1,
        llm_call=LLMCall(
            messages=None,
            response_text="Sorry, failed.",
            tool_calls=[],
            timestamp=2.0,
        ),
        tool_results=[],
        duration_ms=30,
    )
    trace = Trace(trace_id="t2", name="test_err", turns=[t0, t1], started_at=1.0, ended_at=3.0)
    output = format_stack_trace(trace, assertion_msg="tool error test")
    assert "error" in output.lower() or "timeout" in output.lower()


def test_format_plain_text_no_ansi() -> None:
    trace = _make_3turn_trace()
    output = format_stack_trace(trace, assertion_msg="test", color=False)
    assert "\033[" not in output
