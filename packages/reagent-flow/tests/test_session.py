"""Tests for reagent_flow session and recorder."""

import warnings

import pytest
import reagent_flow
from reagent_flow._context import get_active_session
from reagent_flow.exceptions import AmbiguousToolCallError, ReagentWarning, SessionClosedError


def test_session_sets_contextvar() -> None:
    assert get_active_session() is None
    with reagent_flow.session("test") as s:
        assert get_active_session() is s
    assert get_active_session() is None


def test_session_log_llm_call() -> None:
    with reagent_flow.session("test") as s:
        call_ids = s.log_llm_call(
            response_text=None,
            tool_calls=[{"name": "lookup", "arguments": {"id": "1"}}],
        )
    assert len(call_ids) == 1
    assert len(s.trace.turns) == 1
    assert s.trace.turns[0].llm_call.tool_calls[0].name == "lookup"


def test_session_log_tool_result() -> None:
    with reagent_flow.session("test") as s:
        s.log_llm_call(
            response_text=None,
            tool_calls=[{"name": "lookup", "arguments": {"id": "1"}}],
        )
        s.log_tool_result("lookup", result={"found": True}, duration_ms=50.0)
    assert len(s.trace.turns[0].tool_results) == 1
    assert s.trace.turns[0].tool_results[0].result == {"found": True}


def test_session_log_tool_result_by_call_id() -> None:
    with reagent_flow.session("test") as s:
        ids = s.log_llm_call(
            tool_calls=[
                {"name": "lookup", "arguments": {"id": "1"}},
                {"name": "lookup", "arguments": {"id": "2"}},
            ],
        )
        s.log_tool_result("lookup", call_id=ids[0], result={"id": "1"})
        s.log_tool_result("lookup", call_id=ids[1], result={"id": "2"})
    results = s.trace.turns[0].tool_results
    assert results[0].result == {"id": "1"}
    assert results[1].result == {"id": "2"}


def test_session_ambiguous_tool_call_error() -> None:
    with reagent_flow.session("test") as s:
        s.log_llm_call(
            tool_calls=[
                {"name": "lookup", "arguments": {"id": "1"}},
                {"name": "lookup", "arguments": {"id": "2"}},
            ],
        )
        with pytest.raises(AmbiguousToolCallError):
            s.log_tool_result("lookup", result={"found": True})


def test_session_closed_error() -> None:
    with reagent_flow.session("test") as s:
        pass
    with pytest.raises(SessionClosedError):
        s.log_llm_call(response_text="hello", tool_calls=[])


def test_session_text_only_turn() -> None:
    with reagent_flow.session("test") as s:
        s.log_llm_call(response_text="I can help with that", tool_calls=[])
    assert s.trace.turns[0].llm_call.response_text == "I can help with that"
    assert s.trace.turns[0].llm_call.tool_calls == []


def test_session_multiple_turns() -> None:
    with reagent_flow.session("test") as s:
        s.log_llm_call(tool_calls=[{"name": "a", "arguments": {}}])
        s.log_tool_result("a", result="ok")
        s.log_llm_call(tool_calls=[{"name": "b", "arguments": {}}])
        s.log_tool_result("b", result="ok")
        s.log_llm_call(response_text="done", tool_calls=[])
    assert len(s.trace.turns) == 3


def test_session_trace_metadata() -> None:
    with reagent_flow.session("test", metadata={"env": "ci"}) as s:
        pass
    assert s.trace.metadata == {"env": "ci"}


def test_session_assert_called_inside_active_session() -> None:
    """Assertions should work while the session is still active."""
    with reagent_flow.session("test") as s:
        s.log_llm_call(tool_calls=[{"name": "lookup", "arguments": {}}])
        s.log_tool_result("lookup", result="ok")
        s.assert_called("lookup")
        s.assert_never_called("delete")
        s.assert_max_turns(5)


def test_log_tool_result_before_llm_call_warns() -> None:
    """log_tool_result before any log_llm_call should warn, not silently drop."""
    with reagent_flow.session("test") as s:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            s.log_tool_result("lookup", result="ok")
    assert any(issubclass(w.category, ReagentWarning) for w in caught)
    assert len(s.trace.turns) == 0


def test_session_assert_flow() -> None:
    """assert_flow wrapper works on Session."""
    with reagent_flow.session("test") as s:
        s.log_llm_call(tool_calls=[{"name": "search", "arguments": {}}])
        s.log_tool_result("search", result="ok")
        s.log_llm_call(tool_calls=[{"name": "summarize", "arguments": {}}])
        s.log_tool_result("summarize", result="ok")
    s.assert_flow(["search", ..., "summarize"])


def test_session_assert_called_times() -> None:
    """assert_called_times wrapper works on Session."""
    with reagent_flow.session("test") as s:
        s.log_llm_call(tool_calls=[{"name": "search", "arguments": {}}])
        s.log_tool_result("search", result="ok")
        s.log_llm_call(tool_calls=[{"name": "search", "arguments": {}}])
        s.log_tool_result("search", result="ok")
    s.assert_called_times("search", min=1, max=3)


def test_session_assert_called_with() -> None:
    """assert_called_with wrapper works on Session."""
    with reagent_flow.session("test") as s:
        s.log_llm_call(tool_calls=[{"name": "search", "arguments": {"query": "test"}}])
        s.log_tool_result("search", result="ok")
    s.assert_called_with("search", query="test")


def test_session_handoff_integration() -> None:
    """Full handoff flow: parent -> child with context."""
    with reagent_flow.session("orchestrator") as parent:
        parent.log_llm_call(tool_calls=[{"name": "plan", "arguments": {}}])
        parent.log_tool_result("plan", result="ok")

    with reagent_flow.session(
        "researcher",
        parent_trace_id=parent.trace.trace_id,
        handoff_context={"query": "Q3 earnings", "constraints": ["2024"]},
    ) as child:
        child.log_llm_call(tool_calls=[{"name": "search", "arguments": {}}])
        child.log_tool_result("search", result="ok")

    child.assert_handoff_received(parent)
    child.assert_handoff_has_fields(["query", "constraints"])


def test_session_assert_total_tokens_under() -> None:
    """assert_total_tokens_under wrapper works on Session."""
    with reagent_flow.session("test") as s:
        s.log_llm_call(
            tool_calls=[{"name": "search", "arguments": {}}],
            token_usage={"input_tokens": 100, "output_tokens": 200},
            model="gpt-4o",
        )
        s.log_tool_result("search", result="ok")
    s.assert_total_tokens_under(500)


def test_session_assert_cost_under() -> None:
    """assert_cost_under wrapper works on Session."""
    with reagent_flow.session("test") as s:
        s.log_llm_call(
            tool_calls=[{"name": "search", "arguments": {}}],
            token_usage={"input_tokens": 100, "output_tokens": 50},
            model="gpt-4o",
        )
        s.log_tool_result("search", result="ok")
    s.assert_cost_under(usd=1.00, model_costs={"gpt-4o": {"input": 2.50, "output": 10.00}})


@pytest.mark.asyncio
async def test_async_session_context_manager() -> None:
    """Session should work as an async context manager."""
    async with reagent_flow.session("async_test") as s:
        s.log_llm_call(tool_calls=[{"name": "search", "arguments": {"q": "test"}}])
        s.log_tool_result("search", result={"found": True})
        assert get_active_session() is s
    assert get_active_session() is None
    assert len(s.trace.turns) == 1
    assert s.trace.ended_at is not None
