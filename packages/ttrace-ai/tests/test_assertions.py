"""Tests for assertion engine."""

import pytest
import ttrace_ai


def _session_with_tools(*tool_names: str) -> ttrace_ai.Session:
    """Create a session with sequential tool calls for testing assertions."""
    s = ttrace_ai.session("test")
    s.__enter__()
    for name in tool_names:
        s.log_llm_call(tool_calls=[{"name": name, "arguments": {}}])
        s.log_tool_result(name, result="ok")
    s.log_llm_call(response_text="done", tool_calls=[])
    s.__exit__(None, None, None)
    return s


def test_assert_called_pass() -> None:
    s = _session_with_tools("lookup", "process")
    s.assert_called("lookup")
    s.assert_called("process")


def test_assert_called_fail() -> None:
    s = _session_with_tools("lookup")
    with pytest.raises(AssertionError, match="never called"):
        s.assert_called("process")


def test_assert_never_called_pass() -> None:
    s = _session_with_tools("lookup")
    s.assert_never_called("delete")


def test_assert_never_called_fail() -> None:
    s = _session_with_tools("lookup")
    with pytest.raises(AssertionError, match="was called"):
        s.assert_never_called("lookup")


def test_assert_called_before_pass() -> None:
    s = _session_with_tools("lookup", "process")
    s.assert_called_before("lookup", "process")


def test_assert_called_before_fail() -> None:
    s = _session_with_tools("lookup", "process")
    with pytest.raises(AssertionError):
        s.assert_called_before("process", "lookup")


def test_assert_tool_succeeded_pass() -> None:
    s = _session_with_tools("lookup")
    s.assert_tool_succeeded("lookup")


def test_assert_tool_succeeded_fail() -> None:
    s = ttrace_ai.session("test")
    s.__enter__()
    s.log_llm_call(tool_calls=[{"name": "lookup", "arguments": {}}])
    s.log_tool_result("lookup", error="not found")
    s.__exit__(None, None, None)
    with pytest.raises(AssertionError, match="failed"):
        s.assert_tool_succeeded("lookup")


def test_assert_max_turns_pass() -> None:
    s = _session_with_tools("a", "b")
    s.assert_max_turns(5)


def test_assert_max_turns_fail() -> None:
    s = _session_with_tools("a", "b", "c")
    with pytest.raises(AssertionError, match="turns"):
        s.assert_max_turns(2)


def test_assert_total_duration_under_pass() -> None:
    s = _session_with_tools("a")
    s.assert_total_duration_under(ms=999999)


def test_assert_total_duration_under_fail() -> None:
    s = _session_with_tools("a")
    s.trace.started_at = 0
    s.trace.ended_at = 10.0
    with pytest.raises(AssertionError, match="duration"):
        s.assert_total_duration_under(ms=1)


def test_assert_tool_succeeded_missing_result() -> None:
    """assert_tool_succeeded should fail when a tool call has no recorded result."""
    s = ttrace_ai.session("test")
    s.__enter__()
    s.log_llm_call(tool_calls=[{"name": "lookup", "arguments": {}}])
    # Deliberately do NOT log a tool result
    s.__exit__(None, None, None)
    with pytest.raises(AssertionError, match="no recorded result"):
        s.assert_tool_succeeded("lookup")
