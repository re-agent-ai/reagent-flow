"""Tests for golden baseline diff engine."""

from reagent_flow.diff import TurnDiff, diff_traces
from reagent_flow.models import LLMCall, ToolCall, ToolResult, Trace, Turn


def _make_turn(index: int, tool_name: str, args: dict[str, object] | None = None) -> Turn:
    tc = ToolCall(name=tool_name, arguments=args or {}, call_id=f"c{index}", timestamp=1.0)
    tr = ToolResult(call_id=f"c{index}", result="ok")
    lc = LLMCall(messages=None, response_text=None, tool_calls=[tc], timestamp=1.0)
    return Turn(index=index, llm_call=lc, tool_results=[tr])


def _make_trace(
    name: str,
    tools: list[str],
    args: dict[str, object] | None = None,
) -> Trace:
    turns = [_make_turn(i, t, args) for i, t in enumerate(tools)]
    return Trace(trace_id="t1", name=name, turns=turns)


def test_identical_traces_match() -> None:
    golden = _make_trace("test", ["lookup", "process", "respond"])
    actual = _make_trace("test", ["lookup", "process", "respond"])
    result = diff_traces(golden, actual)
    assert result.is_match
    assert "match" in result.summary().lower()


def test_different_tool_names() -> None:
    golden = _make_trace("test", ["lookup", "process"])
    actual = _make_trace("test", ["lookup", "delete"])
    result = diff_traces(golden, actual)
    assert not result.is_match
    assert result.turn_diffs[1].tool_name_match is False


def test_extra_turns() -> None:
    golden = _make_trace("test", ["lookup"])
    actual = _make_trace("test", ["lookup", "extra"])
    result = diff_traces(golden, actual)
    assert not result.is_match
    assert result.turn_diffs[1].extra is True


def test_missing_turns() -> None:
    golden = _make_trace("test", ["lookup", "process"])
    actual = _make_trace("test", ["lookup"])
    result = diff_traces(golden, actual)
    assert not result.is_match
    assert result.turn_diffs[1].missing is True


def test_argument_diffs() -> None:
    golden = _make_trace("test", ["lookup"])
    golden.turns[0].llm_call.tool_calls[0].arguments = {"id": "1"}
    actual = _make_trace("test", ["lookup"])
    actual.turns[0].llm_call.tool_calls[0].arguments = {"id": "2"}
    result = diff_traces(golden, actual)
    assert not result.is_match
    assert "lookup.id" in result.turn_diffs[0].argument_diffs


def test_diff_summary_readable() -> None:
    golden = _make_trace("test", ["lookup", "process"])
    actual = _make_trace("test", ["lookup", "delete"])
    result = diff_traces(golden, actual)
    summary = result.summary()
    assert "mismatch" in summary.lower()


def test_turn_diff_is_match_property() -> None:
    td = TurnDiff(turn_index=0)
    assert td.is_match is True
    td.tool_name_match = False
    assert td.is_match is False


def test_result_mismatch_detected() -> None:
    """Diff should detect when tool results differ between traces."""
    golden = _make_trace("test", ["lookup"])
    actual = _make_trace("test", ["lookup"])
    actual.turns[0].tool_results[0].result = "different"
    result = diff_traces(golden, actual)
    assert not result.is_match
    assert result.turn_diffs[0].result_match is False


def test_response_text_mismatch_detected() -> None:
    """Diff should detect when response text differs between traces."""
    golden = _make_trace("test", ["lookup"])
    golden.turns[0].llm_call.response_text = "answer A"
    actual = _make_trace("test", ["lookup"])
    actual.turns[0].llm_call.response_text = "answer B"
    result = diff_traces(golden, actual)
    assert not result.is_match
    assert result.turn_diffs[0].result_match is False


def test_result_count_mismatch_detected() -> None:
    """Diff should detect when number of tool results differs."""
    golden = _make_trace("test", ["lookup"])
    actual = _make_trace("test", ["lookup"])
    actual.turns[0].tool_results = []  # remove the result
    result = diff_traces(golden, actual)
    assert not result.is_match
    assert result.turn_diffs[0].result_match is False


def _make_multi_tool_turn(index: int, tool_names: list[str]) -> Turn:
    """Create a turn with multiple parallel tool calls."""
    tcs = [
        ToolCall(name=name, arguments={}, call_id=f"c{index}_{j}", timestamp=1.0)
        for j, name in enumerate(tool_names)
    ]
    trs = [ToolResult(call_id=tc.call_id, result="ok") for tc in tcs]
    lc = LLMCall(messages=None, response_text=None, tool_calls=tcs, timestamp=1.0)
    return Turn(index=index, llm_call=lc, tool_results=trs)


def test_multi_tool_calls_compared() -> None:
    """Diff should compare all tool calls in a turn, not just the first."""
    golden = Trace(
        trace_id="t1",
        name="test",
        turns=[_make_multi_tool_turn(0, ["lookup", "validate"])],
    )
    actual = Trace(
        trace_id="t2",
        name="test",
        turns=[_make_multi_tool_turn(0, ["lookup", "delete"])],
    )
    result = diff_traces(golden, actual)
    assert not result.is_match
    assert result.turn_diffs[0].tool_name_match is False


def test_multi_tool_calls_match() -> None:
    """Identical multi-tool turns should match."""
    golden = Trace(
        trace_id="t1",
        name="test",
        turns=[_make_multi_tool_turn(0, ["lookup", "validate"])],
    )
    actual = Trace(
        trace_id="t2",
        name="test",
        turns=[_make_multi_tool_turn(0, ["lookup", "validate"])],
    )
    result = diff_traces(golden, actual)
    assert result.is_match


def test_tool_count_mismatch_detected() -> None:
    """Different number of tool calls in a turn should be detected."""
    golden = Trace(
        trace_id="t1",
        name="test",
        turns=[_make_multi_tool_turn(0, ["lookup"])],
    )
    actual = Trace(
        trace_id="t2",
        name="test",
        turns=[_make_multi_tool_turn(0, ["lookup", "extra"])],
    )
    result = diff_traces(golden, actual)
    assert not result.is_match
    assert result.turn_diffs[0].tool_name_match is False


def test_result_comparison_positional() -> None:
    """Results should be compared by position, not by call_id."""
    golden = _make_trace("test", ["lookup"])
    actual = _make_trace("test", ["lookup"])
    # Different call_ids but same result — should match
    actual.turns[0].tool_results[0].call_id = "completely_different_id"
    result = diff_traces(golden, actual)
    assert result.is_match


def test_ignore_fields_arguments() -> None:
    """ignore_fields={'arguments'} should skip all argument comparison."""
    golden = _make_trace("test", ["lookup"], args={"id": "1"})
    actual = _make_trace("test", ["lookup"], args={"id": "999"})
    result = diff_traces(golden, actual, ignore_fields={"arguments"})
    assert result.is_match


def test_ignore_fields_results() -> None:
    """ignore_fields={'results'} should skip all result comparison."""
    golden = _make_trace("test", ["lookup"])
    actual = _make_trace("test", ["lookup"])
    actual.turns[0].tool_results[0].result = "different"
    result = diff_traces(golden, actual, ignore_fields={"results"})
    assert result.is_match


def test_ignore_fields_response_text() -> None:
    """ignore_fields={'response_text'} should skip response text comparison."""
    golden = _make_trace("test", ["lookup"])
    actual = _make_trace("test", ["lookup"])
    golden.turns[0].llm_call.response_text = "foo"
    actual.turns[0].llm_call.response_text = "bar"
    result = diff_traces(golden, actual, ignore_fields={"response_text"})
    assert result.is_match


def test_ignore_fields_specific_arg_key() -> None:
    """ignore_fields={'tool.key'} should skip just that one argument."""
    golden = _make_trace("test", ["lookup"], args={"id": "1", "ts": "old"})
    actual = _make_trace("test", ["lookup"], args={"id": "1", "ts": "new"})
    result = diff_traces(golden, actual, ignore_fields={"lookup.ts"})
    assert result.is_match


def test_ignore_fields_does_not_mask_tool_name_mismatch() -> None:
    """ignore_fields should not suppress tool name differences."""
    golden = _make_trace("test", ["lookup"])
    actual = _make_trace("test", ["delete"])
    result = diff_traces(golden, actual, ignore_fields={"arguments", "results"})
    assert not result.is_match
