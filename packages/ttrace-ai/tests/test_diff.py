"""Tests for golden baseline diff engine."""


from ttrace_ai.diff import TurnDiff, diff_traces
from ttrace_ai.models import LLMCall, ToolCall, ToolResult, Trace, Turn


def _make_turn(index: int, tool_name: str, args: dict[str, object] | None = None) -> Turn:
    tc = ToolCall(name=tool_name, arguments=args or {}, call_id=f"c{index}", timestamp=1.0)
    tr = ToolResult(call_id=f"c{index}", result="ok")
    lc = LLMCall(messages=None, response_text=None, tool_calls=[tc], timestamp=1.0)
    return Turn(index=index, llm_call=lc, tool_results=[tr])


def _make_trace(name: str, tools: list[str]) -> Trace:
    turns = [_make_turn(i, t) for i, t in enumerate(tools)]
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
    assert "id" in result.turn_diffs[0].argument_diffs


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
