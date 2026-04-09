"""Tests for assertion engine."""

import warnings
from typing import Any

import pytest
import reagent_flow
from reagent_flow.assertions import (
    assert_called_times,
    assert_called_with,
    assert_context_preserved,
    assert_cost_under,
    assert_flow,
    assert_handoff_has_fields,
    assert_handoff_matches,
    assert_handoff_received,
    assert_no_extra_fields,
    assert_tool_output_matches,
    assert_total_tokens_under,
)
from reagent_flow.exceptions import ReagentWarning
from reagent_flow.models import LLMCall, Trace, Turn


def _session_with_tools(*tool_names: str) -> reagent_flow.Session:
    """Create a session with sequential tool calls for testing assertions."""
    s = reagent_flow.session("test")
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


def test_assert_called_before_same_turn() -> None:
    """Parallel calls in a single turn should respect list position order."""
    s = _session_with_calls(["a", "b"])
    s.assert_called_before("a", "b")


def test_assert_called_before_same_turn_wrong_order() -> None:
    """Parallel calls in wrong order within a single turn should fail."""
    s = _session_with_calls(["b", "a"])
    with pytest.raises(AssertionError):
        s.assert_called_before("a", "b")


def test_assert_tool_succeeded_pass() -> None:
    s = _session_with_tools("lookup")
    s.assert_tool_succeeded("lookup")


def test_assert_tool_succeeded_fail() -> None:
    s = reagent_flow.session("test")
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
    s = reagent_flow.session("test")
    s.__enter__()
    s.log_llm_call(tool_calls=[{"name": "lookup", "arguments": {}}])
    # Deliberately do NOT log a tool result
    s.__exit__(None, None, None)
    with pytest.raises(AssertionError, match="no recorded result"):
        s.assert_tool_succeeded("lookup")


def test_assert_total_duration_under_works_in_active_session() -> None:
    """assert_total_duration_under should use wall clock when ended_at is None."""
    with reagent_flow.session("test") as s:
        s.log_llm_call(tool_calls=[{"name": "a", "arguments": {}}])
        s.log_tool_result("a", result="ok")
        # Session is still active (ended_at is None), assertion should work
        s.assert_total_duration_under(ms=999999)


# ---------------------------------------------------------------------------
# assert_flow tests
# ---------------------------------------------------------------------------


def _session_with_calls(*turns: list[str] | str | None) -> reagent_flow.Session:
    """Create a session where each arg is a turn.

    If a turn is a string, it's a single tool call.
    If a turn is a list of strings, they are parallel tool calls in one turn.
    If a turn is None, it's a text-only turn.
    """
    s = reagent_flow.session("test")
    s.__enter__()
    for turn in turns:
        if turn is None:
            s.log_llm_call(response_text="thinking...", tool_calls=[])
        elif isinstance(turn, str):
            s.log_llm_call(tool_calls=[{"name": turn, "arguments": {}}])
            s.log_tool_result(turn, result="ok")
        else:
            tc_list = [{"name": name, "arguments": {}} for name in turn]
            ids = s.log_llm_call(tool_calls=tc_list)
            for name, cid in zip(turn, ids):
                s.log_tool_result(name, call_id=cid, result="ok")
    s.__exit__(None, None, None)
    return s


def test_assert_flow_consecutive() -> None:
    """["search", "summarize"] matches exactly two consecutive calls."""
    s = _session_with_calls("search", "summarize")
    assert_flow(s.trace, ["search", "summarize"])


def test_assert_flow_consecutive_rejects_gap() -> None:
    """["search", "summarize"] fails when there's a call between them."""
    s = _session_with_calls("search", "lookup", "summarize")
    with pytest.raises(AssertionError, match="summarize"):
        assert_flow(s.trace, ["search", "summarize"])


def test_assert_flow_with_gaps() -> None:
    """["search", ..., "summarize"] allows intervening calls."""
    s = _session_with_calls("search", "lookup", "fetch", "summarize")
    assert_flow(s.trace, ["search", ..., "summarize"])


def test_assert_flow_ellipsis_allows_gap() -> None:
    """["search", ..., "summarize"] passes on [search, lookup, summarize]."""
    s = _session_with_calls("search", "lookup", "summarize")
    assert_flow(s.trace, ["search", ..., "summarize"])


def test_assert_flow_fails_wrong_order() -> None:
    """Tools present but in wrong order."""
    s = _session_with_calls("summarize", "search")
    with pytest.raises(AssertionError):
        assert_flow(s.trace, ["search", ..., "summarize"])


def test_assert_flow_fails_missing() -> None:
    """Tool not in trace at all."""
    s = _session_with_calls("search", "lookup")
    with pytest.raises(AssertionError, match="summarize"):
        assert_flow(s.trace, ["search", ..., "summarize"])


def test_assert_flow_empty_pattern() -> None:
    """Empty pattern always passes."""
    s = _session_with_calls("search", "summarize")
    assert_flow(s.trace, [])


def test_assert_flow_leading_ellipsis() -> None:
    """[..., "summarize"] matches regardless of what comes before."""
    s = _session_with_calls("search", "lookup", "summarize")
    assert_flow(s.trace, [..., "summarize"])


def test_assert_flow_trailing_ellipsis() -> None:
    """["search", ...] matches if first call is search, ignores rest."""
    s = _session_with_calls("search", "lookup", "summarize")
    assert_flow(s.trace, ["search", ...])


def test_assert_flow_anchored_start() -> None:
    """["search", ...] fails if first call is not search."""
    s = _session_with_calls("lookup", "search", "summarize")
    with pytest.raises(AssertionError):
        assert_flow(s.trace, ["search", ...])


def test_assert_flow_no_trailing_ellipsis_rejects_extra() -> None:
    """["search"] fails on [search, summarize] — no trailing ... means anchored end."""
    s = _session_with_calls("search", "summarize")
    with pytest.raises(AssertionError):
        assert_flow(s.trace, ["search"])


def test_assert_flow_multiple_ellipsis() -> None:
    """Multiple consecutive ... collapse to one."""
    s = _session_with_calls("a", "x", "b")
    assert_flow(s.trace, ["a", ..., ..., "b"])


def test_assert_flow_only_ellipsis() -> None:
    """Pattern with only ... always passes."""
    s = _session_with_calls("search", "summarize")
    assert_flow(s.trace, [...])


def test_assert_flow_text_only_turns_skipped() -> None:
    """Text-only turns produce no entries in the flat list."""
    s = _session_with_calls("search", None, "summarize")
    assert_flow(s.trace, ["search", "summarize"])


def test_assert_flow_parallel_calls_same_turn() -> None:
    """Parallel calls in one turn are ordered by list position."""
    s = _session_with_calls(["search", "fetch"])
    assert_flow(s.trace, ["search", "fetch"])


def test_assert_flow_invalid_pattern_element() -> None:
    """Non-str/Ellipsis elements raise TypeError."""
    s = _session_with_calls("search")
    with pytest.raises(TypeError, match="Only str and \\.\\.\\. are allowed"):
        assert_flow(s.trace, ["search", 123])


# ---------------------------------------------------------------------------
# assert_called_times tests
# ---------------------------------------------------------------------------


def test_assert_called_times_in_bounds() -> None:
    s = _session_with_tools("search", "search", "summarize")
    assert_called_times(s.trace, "search", min=1, max=3)


def test_assert_called_times_under_min() -> None:
    s = _session_with_tools("search")
    with pytest.raises(AssertionError, match="expected at least 2"):
        assert_called_times(s.trace, "search", min=2)


def test_assert_called_times_over_max() -> None:
    s = _session_with_tools("search", "search", "search")
    with pytest.raises(AssertionError, match="expected at most 2"):
        assert_called_times(s.trace, "search", max=2)


def test_assert_called_times_invalid_bounds() -> None:
    s = _session_with_tools("search")
    with pytest.raises(ValueError, match="max .* less than min"):
        assert_called_times(s.trace, "search", min=5, max=2)


# ---------------------------------------------------------------------------
# assert_called_with tests
# ---------------------------------------------------------------------------


def test_assert_called_with_match() -> None:
    """Partial arg match -- only specified keys checked."""
    s = reagent_flow.session("test")
    s.__enter__()
    s.log_llm_call(tool_calls=[{"name": "search", "arguments": {"query": "earnings", "limit": 10}}])
    s.log_tool_result("search", result="ok")
    s.__exit__(None, None, None)
    assert_called_with(s.trace, "search", query="earnings")


def test_assert_called_with_no_match() -> None:
    s = reagent_flow.session("test")
    s.__enter__()
    s.log_llm_call(tool_calls=[{"name": "search", "arguments": {"query": "revenue"}}])
    s.log_tool_result("search", result="ok")
    s.__exit__(None, None, None)
    with pytest.raises(AssertionError, match="query"):
        assert_called_with(s.trace, "search", query="earnings")


def test_assert_called_with_nested_values() -> None:
    """Dict/list values compared by equality."""
    s = reagent_flow.session("test")
    s.__enter__()
    s.log_llm_call(tool_calls=[{"name": "search", "arguments": {"filters": {"year": 2024}}}])
    s.log_tool_result("search", result="ok")
    s.__exit__(None, None, None)
    assert_called_with(s.trace, "search", filters={"year": 2024})


def test_assert_called_with_closest_mismatch() -> None:
    """Error message shows closest partial match."""
    s = reagent_flow.session("test")
    s.__enter__()
    s.log_llm_call(tool_calls=[{"name": "search", "arguments": {"query": "revenue", "limit": 5}}])
    s.log_tool_result("search", result="ok")
    s.__exit__(None, None, None)
    with pytest.raises(AssertionError, match="query"):
        assert_called_with(s.trace, "search", query="earnings", limit=5)


# ---------------------------------------------------------------------------
# handoff assertion tests
# ---------------------------------------------------------------------------


def test_handoff_received_linked() -> None:
    parent = Trace(trace_id="parent-1", name="orchestrator")
    child = Trace(trace_id="child-1", name="researcher", parent_trace_id="parent-1")
    assert_handoff_received(child, parent)


def test_handoff_received_not_linked() -> None:
    parent = Trace(trace_id="parent-1", name="orchestrator")
    child = Trace(trace_id="child-1", name="researcher")
    with pytest.raises(AssertionError, match="no parent_trace_id"):
        assert_handoff_received(child, parent)


def test_handoff_received_wrong_parent() -> None:
    parent = Trace(trace_id="parent-1", name="orchestrator")
    child = Trace(trace_id="child-1", name="researcher", parent_trace_id="other-parent")
    with pytest.raises(AssertionError, match="does not match"):
        assert_handoff_received(child, parent)


def test_handoff_has_fields_present() -> None:
    child = Trace(
        trace_id="c1",
        name="researcher",
        handoff_context={"query": "test", "constraints": ["2024"]},
    )
    assert_handoff_has_fields(child, fields=["query", "constraints"])


def test_handoff_has_fields_missing() -> None:
    child = Trace(
        trace_id="c1",
        name="researcher",
        handoff_context={"query": "test"},
    )
    with pytest.raises(AssertionError, match="constraints"):
        assert_handoff_has_fields(child, fields=["query", "constraints"])


def test_handoff_has_fields_none_value() -> None:
    child = Trace(
        trace_id="c1",
        name="researcher",
        handoff_context={"query": None},
    )
    with pytest.raises(AssertionError, match="query"):
        assert_handoff_has_fields(child, fields=["query"])


def test_handoff_has_fields_no_context() -> None:
    child = Trace(trace_id="c1", name="researcher")
    with pytest.raises(AssertionError, match="no handoff_context"):
        assert_handoff_has_fields(child, fields=["query"])


def test_handoff_non_dict_context() -> None:
    """handoff_context that is not a dict raises assertion error."""
    child = Trace(
        trace_id="c1",
        name="researcher",
        handoff_context="not a dict",  # type: ignore[arg-type]
    )
    with pytest.raises(AssertionError, match="must be a dict"):
        assert_handoff_has_fields(child, fields=["query"])


# ---------------------------------------------------------------------------
# assert_total_tokens_under tests
# ---------------------------------------------------------------------------


def _trace_with_usage(*usages: dict[str, int] | None) -> Trace:
    """Create a trace with specified token_usage per turn."""
    turns = []
    for i, usage in enumerate(usages):
        llm_call = LLMCall(
            messages=None,
            response_text=None,
            tool_calls=[],
            token_usage=usage,
            model="gpt-4o",
        )
        turns.append(Turn(index=i, llm_call=llm_call, tool_results=[]))
    return Trace(trace_id="t1", name="test", turns=turns)


def test_tokens_under_limit() -> None:
    tr = _trace_with_usage({"input_tokens": 100, "output_tokens": 200})
    assert_total_tokens_under(tr, n=500)


def test_tokens_over_limit() -> None:
    tr = _trace_with_usage({"input_tokens": 300, "output_tokens": 300})
    with pytest.raises(AssertionError, match="600"):
        assert_total_tokens_under(tr, n=500)


def test_tokens_mixed_formats() -> None:
    """OpenAI and Anthropic key names in same trace."""
    turns = [
        Turn(
            index=0,
            llm_call=LLMCall(
                messages=None,
                response_text=None,
                tool_calls=[],
                token_usage={"prompt_tokens": 100, "completion_tokens": 50},
                model="gpt-4o",
            ),
            tool_results=[],
        ),
        Turn(
            index=1,
            llm_call=LLMCall(
                messages=None,
                response_text=None,
                tool_calls=[],
                token_usage={"input_tokens": 100, "output_tokens": 50},
                model="claude-sonnet-4-6",
            ),
            tool_results=[],
        ),
    ]
    tr = Trace(trace_id="t1", name="test", turns=turns)
    assert_total_tokens_under(tr, n=400)


def test_tokens_missing_usage_warns() -> None:
    tr = _trace_with_usage({"input_tokens": 100, "output_tokens": 50}, None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert_total_tokens_under(tr, n=500)
    assert any(issubclass(w.category, ReagentWarning) for w in caught)


def test_tokens_all_missing_fails() -> None:
    """All turns missing token_usage — assertion fails by default."""
    tr = _trace_with_usage(None, None)
    with pytest.raises(AssertionError, match="no token data"):
        assert_total_tokens_under(tr, n=500)


def test_tokens_all_missing_allow() -> None:
    """allow_missing=True: passes with warnings when all missing."""
    tr = _trace_with_usage(None, None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert_total_tokens_under(tr, n=500, allow_missing=True)
    assert any(issubclass(w.category, ReagentWarning) for w in caught)


def test_tokens_zero_value_counted() -> None:
    """0 is a valid token count, not treated as falsy."""
    tr = _trace_with_usage({"input_tokens": 0, "output_tokens": 500})
    assert_total_tokens_under(tr, n=501)
    with pytest.raises(AssertionError):
        assert_total_tokens_under(tr, n=499)


# ---------------------------------------------------------------------------
# assert_cost_under tests
# ---------------------------------------------------------------------------


def _trace_with_models(*model_usages: tuple[str | None, dict[str, int] | None]) -> Trace:
    """Create a trace with specified model + token_usage per turn."""
    turns = []
    for i, (model, usage) in enumerate(model_usages):
        llm_call = LLMCall(
            messages=None,
            response_text=None,
            tool_calls=[],
            token_usage=usage,
            model=model,
        )
        turns.append(Turn(index=i, llm_call=llm_call, tool_results=[]))
    return Trace(trace_id="t1", name="test", turns=turns)


COSTS = {"gpt-4o": {"input": 2.50, "output": 10.00}}


def test_cost_under_limit() -> None:
    tr = _trace_with_models(("gpt-4o", {"input_tokens": 100, "output_tokens": 50}))
    assert_cost_under(tr, usd=0.01, model_costs=COSTS)


def test_cost_over_limit() -> None:
    tr = _trace_with_models(("gpt-4o", {"input_tokens": 1_000_000, "output_tokens": 1_000_000}))
    with pytest.raises(AssertionError, match="12.50"):
        assert_cost_under(tr, usd=1.00, model_costs=COSTS)


def test_cost_prefix_matching() -> None:
    """'gpt-4o' in costs matches 'gpt-4o-2024-08-06' in trace."""
    tr = _trace_with_models(("gpt-4o-2024-08-06", {"input_tokens": 100, "output_tokens": 50}))
    assert_cost_under(tr, usd=0.01, model_costs=COSTS)


def test_cost_longest_prefix_wins() -> None:
    """'gpt-4o-mini' preferred over 'gpt-4o' for model 'gpt-4o-mini-2024'."""
    costs = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    }
    tr = _trace_with_models(
        (
            "gpt-4o-mini-2024-07-18",
            {"input_tokens": 1_000_000, "output_tokens": 1_000_000},
        )
    )
    assert_cost_under(tr, usd=1.00, model_costs=costs)


def test_cost_unknown_model_warns() -> None:
    tr = _trace_with_models(("unknown-model", {"input_tokens": 100, "output_tokens": 50}))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(AssertionError, match="no priced turns"):
            assert_cost_under(tr, usd=0.01, model_costs=COSTS)
    assert any("unknown-model" in str(w.message) for w in caught)


def test_cost_missing_model_warns() -> None:
    tr = _trace_with_models((None, {"input_tokens": 100, "output_tokens": 50}))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(AssertionError, match="no priced turns"):
            assert_cost_under(tr, usd=0.01, model_costs=COSTS)
    assert any(issubclass(w.category, ReagentWarning) for w in caught)


def test_cost_mixed_known_unknown() -> None:
    """Known models counted, unknown models warned."""
    tr = _trace_with_models(
        ("gpt-4o", {"input_tokens": 1_000_000, "output_tokens": 1_000_000}),
        ("unknown", {"input_tokens": 100, "output_tokens": 50}),
    )
    with pytest.raises(AssertionError):
        assert_cost_under(tr, usd=1.00, model_costs=COSTS)


def test_cost_all_unpriced_fails() -> None:
    """Zero priced turns -> fails by default."""
    tr = _trace_with_models(("unknown", {"input_tokens": 100, "output_tokens": 50}))
    with pytest.raises(AssertionError, match="no priced turns"):
        assert_cost_under(tr, usd=0.01, model_costs=COSTS)


def test_cost_all_unpriced_allow() -> None:
    """allow_unpriced=True: passes with warnings."""
    tr = _trace_with_models(("unknown", {"input_tokens": 100, "output_tokens": 50}))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert_cost_under(tr, usd=0.01, model_costs=COSTS, allow_unpriced=True)
    assert any(issubclass(w.category, ReagentWarning) for w in caught)


# ---------------------------------------------------------------------------
# assert_handoff_matches tests (v0.3)
# ---------------------------------------------------------------------------


def test_handoff_matches_valid() -> None:
    child = Trace(
        trace_id="c1",
        name="researcher",
        handoff_context={"user_id": "abc", "query": "test", "limit": 5},
    )
    assert_handoff_matches(child, schema={"user_id": str, "query": str, "limit": int})


def test_handoff_matches_missing_field() -> None:
    child = Trace(
        trace_id="c1",
        name="researcher",
        handoff_context={"user_id": "abc"},
    )
    with pytest.raises(AssertionError, match="query"):
        assert_handoff_matches(child, schema={"user_id": str, "query": str})


def test_handoff_matches_wrong_type() -> None:
    child = Trace(
        trace_id="c1",
        name="researcher",
        handoff_context={"user_id": 123, "query": "test"},
    )
    with pytest.raises(AssertionError, match="expected str, got int"):
        assert_handoff_matches(child, schema={"user_id": str, "query": str})


def test_handoff_matches_none_context() -> None:
    child = Trace(trace_id="c1", name="researcher")
    with pytest.raises(AssertionError, match="no handoff_context"):
        assert_handoff_matches(child, schema={"user_id": str})


def test_handoff_matches_non_dict_context() -> None:
    child = Trace(
        trace_id="c1",
        name="researcher",
        handoff_context="not a dict",  # type: ignore[arg-type]
    )
    with pytest.raises(AssertionError, match="must be a dict"):
        assert_handoff_matches(child, schema={"user_id": str})


def test_handoff_matches_empty_schema() -> None:
    """Empty schema should always pass."""
    child = Trace(
        trace_id="c1",
        name="researcher",
        handoff_context={"user_id": "abc"},
    )
    assert_handoff_matches(child, schema={})


def test_handoff_matches_bool_not_accepted_as_int() -> None:
    """bool subclasses int in Python, but contracts should distinguish them."""
    child = Trace(
        trace_id="c1",
        name="researcher",
        handoff_context={"count": True},
    )
    with pytest.raises(AssertionError, match="expected int, got bool"):
        assert_handoff_matches(child, schema={"count": int})


def test_handoff_matches_int_not_accepted_as_bool() -> None:
    """int should not satisfy a bool contract."""
    child = Trace(
        trace_id="c1",
        name="researcher",
        handoff_context={"active": 1},
    )
    with pytest.raises(AssertionError, match="expected bool, got int"):
        assert_handoff_matches(child, schema={"active": bool})


# ---------------------------------------------------------------------------
# assert_no_extra_fields tests (v0.3)
# ---------------------------------------------------------------------------


def test_no_extra_fields_all_allowed() -> None:
    child = Trace(
        trace_id="c1",
        name="researcher",
        handoff_context={"user_id": "abc", "query": "test"},
    )
    assert_no_extra_fields(child, allowed=["user_id", "query"])


def test_no_extra_fields_extra_present() -> None:
    child = Trace(
        trace_id="c1",
        name="researcher",
        handoff_context={"user_id": "abc", "debug_info": "x", "internal_id": "y"},
    )
    with pytest.raises(AssertionError, match="debug_info"):
        assert_no_extra_fields(child, allowed=["user_id"])


def test_no_extra_fields_reports_all_extras() -> None:
    """Error should report ALL unexpected keys, not just the first."""
    child = Trace(
        trace_id="c1",
        name="researcher",
        handoff_context={"user_id": "abc", "debug_info": "x", "internal_id": "y"},
    )
    with pytest.raises(AssertionError, match="internal_id"):
        assert_no_extra_fields(child, allowed=["user_id"])


def test_no_extra_fields_none_context() -> None:
    child = Trace(trace_id="c1", name="researcher")
    with pytest.raises(AssertionError, match="no handoff_context"):
        assert_no_extra_fields(child, allowed=["user_id"])


def test_no_extra_fields_empty_allowed_nonempty_context() -> None:
    child = Trace(
        trace_id="c1",
        name="researcher",
        handoff_context={"user_id": "abc"},
    )
    with pytest.raises(AssertionError, match="user_id"):
        assert_no_extra_fields(child, allowed=[])


# ---------------------------------------------------------------------------
# assert_tool_output_matches tests (v0.3)
# ---------------------------------------------------------------------------


def _session_with_tool_results(
    *calls: tuple[str, dict[str, Any] | None, str | None],
) -> reagent_flow.Session:
    """Create a session with tool calls that have specific results.

    Each call is (tool_name, result_dict_or_None, error_or_None).
    """
    s = reagent_flow.session("test")
    s.__enter__()
    for name, result, error in calls:
        ids = s.log_llm_call(tool_calls=[{"name": name, "arguments": {}}])
        s.log_tool_result(
            name,
            call_id=ids[0],
            result=result,
            **({"error": error} if error else {}),
        )
    s.__exit__(None, None, None)
    return s


def test_tool_output_matches_single_call() -> None:
    s = _session_with_tool_results(("search", {"results": ["a"], "count": 1}, None))
    assert_tool_output_matches(s.trace, "search", schema={"results": list, "count": int})


def test_tool_output_matches_wrong_type() -> None:
    s = _session_with_tool_results(("search", {"results": "not a list", "count": 1}, None))
    with pytest.raises(AssertionError, match="expected list, got str"):
        assert_tool_output_matches(s.trace, "search", schema={"results": list, "count": int})


def test_tool_output_matches_multiple_calls_all_match() -> None:
    s = _session_with_tool_results(
        ("search", {"results": ["a"], "count": 1}, None),
        ("search", {"results": ["b"], "count": 2}, None),
    )
    assert_tool_output_matches(s.trace, "search", schema={"results": list, "count": int})


def test_tool_output_matches_multiple_calls_one_fails() -> None:
    s = _session_with_tool_results(
        ("search", {"results": ["a"], "count": 1}, None),
        ("search", {"results": "bad", "count": 2}, None),
    )
    with pytest.raises(AssertionError, match="expected list, got str"):
        assert_tool_output_matches(s.trace, "search", schema={"results": list, "count": int})


def test_tool_output_matches_tool_never_called() -> None:
    s = _session_with_tool_results(("lookup", {"id": "1"}, None))
    with pytest.raises(AssertionError, match="never called"):
        assert_tool_output_matches(s.trace, "search", schema={"results": list})


def test_tool_output_matches_result_not_dict() -> None:
    s = _session_with_tool_results(("search", "plain string", None))
    with pytest.raises(AssertionError, match="not a dict"):
        assert_tool_output_matches(s.trace, "search", schema={"results": list})


def test_tool_output_matches_errored_call_skipped() -> None:
    """Errored calls should be skipped — only successful results validated."""
    s = _session_with_tool_results(
        ("search", None, "timeout"),
        ("search", {"results": ["a"], "count": 1}, None),
    )
    assert_tool_output_matches(s.trace, "search", schema={"results": list, "count": int})


def test_tool_output_matches_missing_field() -> None:
    s = _session_with_tool_results(("search", {"results": ["a"]}, None))
    with pytest.raises(AssertionError, match="count"):
        assert_tool_output_matches(s.trace, "search", schema={"results": list, "count": int})


# ---------------------------------------------------------------------------
# assert_context_preserved tests (v0.3)
# ---------------------------------------------------------------------------


def test_context_preserved_all_match() -> None:
    source = {"user_id": "abc123", "query": "revenue Q4", "org": "acme"}
    child = Trace(
        trace_id="c1",
        name="researcher",
        handoff_context={"user_id": "abc123", "query": "revenue Q4", "extra": "ok"},
    )
    assert_context_preserved(source, child, fields=["user_id", "query"])


def test_context_preserved_value_changed() -> None:
    source = {"user_id": "abc123"}
    child = Trace(
        trace_id="c1",
        name="researcher",
        handoff_context={"user_id": "xyz789"},
    )
    with pytest.raises(AssertionError, match="not preserved"):
        assert_context_preserved(source, child, fields=["user_id"])


def test_context_preserved_missing_in_source() -> None:
    source = {"query": "test"}
    child = Trace(
        trace_id="c1",
        name="researcher",
        handoff_context={"user_id": "abc", "query": "test"},
    )
    with pytest.raises(AssertionError, match="user_id"):
        assert_context_preserved(source, child, fields=["user_id"])


def test_context_preserved_missing_in_handoff() -> None:
    source = {"user_id": "abc123", "query": "test"}
    child = Trace(
        trace_id="c1",
        name="researcher",
        handoff_context={"query": "test"},
    )
    with pytest.raises(AssertionError, match="user_id"):
        assert_context_preserved(source, child, fields=["user_id"])


def test_context_preserved_none_handoff() -> None:
    source = {"user_id": "abc123"}
    child = Trace(trace_id="c1", name="researcher")
    with pytest.raises(AssertionError, match="no handoff_context"):
        assert_context_preserved(source, child, fields=["user_id"])


# ---------------------------------------------------------------------------
# v0.4 nested schema validation tests
# ---------------------------------------------------------------------------


def test_handoff_matches_nested_dict() -> None:
    child = Trace(
        trace_id="c1",
        name="researcher",
        handoff_context={"user": {"id": "abc", "name": "Alice"}, "query": "test"},
    )
    assert_handoff_matches(child, schema={"user": {"id": str, "name": str}, "query": str})


def test_handoff_matches_nested_dict_wrong_type() -> None:
    child = Trace(
        trace_id="c1",
        name="researcher",
        handoff_context={"user": {"id": "abc", "name": 123}},
    )
    with pytest.raises(AssertionError, match="user.name"):
        assert_handoff_matches(child, schema={"user": {"id": str, "name": str}})


def test_handoff_matches_typed_list() -> None:
    child = Trace(
        trace_id="c1",
        name="researcher",
        handoff_context={"tags": ["python", "ai", "ml"]},
    )
    assert_handoff_matches(child, schema={"tags": [str]})


def test_handoff_matches_typed_list_wrong_element() -> None:
    child = Trace(
        trace_id="c1",
        name="researcher",
        handoff_context={"tags": ["python", 42, "ml"]},
    )
    with pytest.raises(AssertionError, match=r"tags\[1\]"):
        assert_handoff_matches(child, schema={"tags": [str]})


def test_handoff_matches_union_list() -> None:
    child = Trace(
        trace_id="c1",
        name="researcher",
        handoff_context={"values": ["hello", 42, "world"]},
    )
    assert_handoff_matches(child, schema={"values": [str, int]})


def test_handoff_matches_union_list_invalid() -> None:
    child = Trace(
        trace_id="c1",
        name="researcher",
        handoff_context={"values": ["hello", 3.14]},
    )
    with pytest.raises(AssertionError, match=r"values\[1\]"):
        assert_handoff_matches(child, schema={"values": [str, int]})


def test_handoff_matches_list_of_dicts() -> None:
    child = Trace(
        trace_id="c1",
        name="researcher",
        handoff_context={
            "results": [
                {"id": "a1", "score": 0.95},
                {"id": "a2", "score": 0.87},
            ]
        },
    )
    assert_handoff_matches(child, schema={"results": [{"id": str, "score": float}]})


def test_handoff_matches_list_of_dicts_invalid() -> None:
    child = Trace(
        trace_id="c1",
        name="researcher",
        handoff_context={
            "results": [
                {"id": "a1", "score": 0.95},
                {"id": "a2", "score": "bad"},
            ]
        },
    )
    with pytest.raises(AssertionError, match=r"results\[1\]\.score"):
        assert_handoff_matches(child, schema={"results": [{"id": str, "score": float}]})


def test_handoff_matches_deep_nesting() -> None:
    """3+ levels of nesting."""
    child = Trace(
        trace_id="c1",
        name="researcher",
        handoff_context={
            "config": {
                "db": {
                    "host": "localhost",
                    "port": 5432,
                },
            },
        },
    )
    assert_handoff_matches(
        child,
        schema={"config": {"db": {"host": str, "port": int}}},
    )


def test_handoff_matches_deep_nesting_error_path() -> None:
    """Error path should use dot notation for deep nesting."""
    child = Trace(
        trace_id="c1",
        name="researcher",
        handoff_context={
            "config": {"db": {"host": "localhost", "port": "bad"}},
        },
    )
    with pytest.raises(AssertionError, match="config.db.port"):
        assert_handoff_matches(
            child,
            schema={"config": {"db": {"host": str, "port": int}}},
        )


def test_handoff_matches_flat_still_works() -> None:
    """v0.3 flat schemas must continue to work unchanged (backward compat)."""
    child = Trace(
        trace_id="c1",
        name="researcher",
        handoff_context={"user_id": "abc", "count": 5},
    )
    assert_handoff_matches(child, schema={"user_id": str, "count": int})


def test_tool_output_matches_nested() -> None:
    """assert_tool_output_matches should also support nested schemas."""
    s = _session_with_tool_results(
        ("search", {"results": [{"id": "a1", "score": 0.9}], "total": 1}, None),
    )
    assert_tool_output_matches(
        s.trace,
        "search",
        schema={"results": [{"id": str, "score": float}], "total": int},
    )


def test_tool_output_matches_nested_invalid() -> None:
    s = _session_with_tool_results(
        ("search", {"results": [{"id": "a1", "score": "bad"}], "total": 1}, None),
    )
    with pytest.raises(AssertionError, match=r"results\[0\]\.score"):
        assert_tool_output_matches(
            s.trace,
            "search",
            schema={"results": [{"id": str, "score": float}], "total": int},
        )


# ---------------------------------------------------------------------------
# v0.4 Pydantic support tests
# ---------------------------------------------------------------------------

_has_pydantic = False
try:
    import pydantic  # noqa: F401

    _has_pydantic = True
except ImportError:
    pass


@pytest.mark.skipif(not _has_pydantic, reason="pydantic not installed")
def test_handoff_matches_pydantic_valid() -> None:
    from pydantic import BaseModel

    class HandoffSchema(BaseModel):
        user_id: str
        query: str

    child = Trace(
        trace_id="c1",
        name="researcher",
        handoff_context={"user_id": "abc", "query": "test"},
    )
    assert_handoff_matches(child, schema=HandoffSchema)  # type: ignore[arg-type]


@pytest.mark.skipif(not _has_pydantic, reason="pydantic not installed")
def test_handoff_matches_pydantic_invalid() -> None:
    from pydantic import BaseModel

    class HandoffSchema(BaseModel):
        user_id: str
        query: str

    child = Trace(
        trace_id="c1",
        name="researcher",
        handoff_context={"user_id": 123, "query": "test"},
    )
    with pytest.raises(AssertionError, match="user_id"):
        assert_handoff_matches(child, schema=HandoffSchema)  # type: ignore[arg-type]


@pytest.mark.skipif(not _has_pydantic, reason="pydantic not installed")
def test_tool_output_matches_pydantic() -> None:
    from pydantic import BaseModel

    class ResultSchema(BaseModel):
        results: list[str]
        count: int

    s = _session_with_tool_results(
        ("search", {"results": ["a", "b"], "count": 2}, None),
    )
    assert_tool_output_matches(s.trace, "search", schema=ResultSchema)  # type: ignore[arg-type]


@pytest.mark.skipif(not _has_pydantic, reason="pydantic not installed")
def test_tool_output_matches_pydantic_invalid() -> None:
    from pydantic import BaseModel

    class ResultSchema(BaseModel):
        results: list[str]
        count: int

    s = _session_with_tool_results(
        ("search", {"results": "not a list", "count": 2}, None),
    )
    with pytest.raises(AssertionError):
        assert_tool_output_matches(s.trace, "search", schema=ResultSchema)  # type: ignore[arg-type]


def test_handoff_matches_no_pydantic_falls_back_to_dict() -> None:
    """When schema is a plain dict, Pydantic code path is never entered.

    This exercises the non-Pydantic path even when pydantic IS installed,
    ensuring the dict-based validation is not accidentally broken.
    """
    child = Trace(
        trace_id="c1",
        name="researcher",
        handoff_context={"user_id": 123, "query": "test"},
    )
    # Even if pydantic is installed, a dict schema must use our validation
    with pytest.raises(AssertionError, match="expected str, got int"):
        assert_handoff_matches(child, schema={"user_id": str, "query": str})
