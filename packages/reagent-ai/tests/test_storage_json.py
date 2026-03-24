"""Tests for JSON trace storage."""

import json

import pytest
from reagent_ai.exceptions import TraceNotFoundError
from reagent_ai.models import LLMCall, ToolCall, ToolResult, Trace, Turn
from reagent_ai.storage.json import find_traces, load_golden, load_trace, save_trace


def _make_trace(name: str = "test_trace") -> Trace:
    tc = ToolCall(name="lookup", arguments={"id": "1"}, call_id="c1", timestamp=1.0)
    tr = ToolResult(call_id="c1", result={"ok": True}, duration_ms=50)
    lc = LLMCall(messages=None, response_text=None, tool_calls=[tc], timestamp=1.0)
    turn = Turn(index=0, llm_call=lc, tool_results=[tr], duration_ms=100)
    return Trace(trace_id="tid", name=name, turns=[turn], started_at=1.0, ended_at=2.0)


def test_save_and_load_trace(tmp_path: object) -> None:
    trace = _make_trace()
    save_trace(trace, str(tmp_path), golden=False)

    traces = find_traces(str(tmp_path), "test_trace")
    assert len(traces) == 1
    loaded = load_trace(traces[0])
    assert loaded.trace_id == "tid"
    assert loaded.turns[0].llm_call.tool_calls[0].name == "lookup"


def test_save_golden_overwrites(tmp_path: object) -> None:
    trace1 = _make_trace()
    trace1.trace_id = "v1"
    save_trace(trace1, str(tmp_path), golden=True)

    trace2 = _make_trace()
    trace2.trace_id = "v2"
    save_trace(trace2, str(tmp_path), golden=True)

    loaded = load_golden(str(tmp_path), "test_trace")
    assert loaded.trace_id == "v2"


def test_load_golden_not_found(tmp_path: object) -> None:
    with pytest.raises(TraceNotFoundError):
        load_golden(str(tmp_path), "nonexistent")


def test_regular_traces_accumulate(tmp_path: object) -> None:
    for i in range(3):
        trace = _make_trace()
        trace.trace_id = f"t{i}"
        save_trace(trace, str(tmp_path), golden=False)

    traces = find_traces(str(tmp_path), "test_trace")
    assert len(traces) == 3


def test_trace_file_is_valid_json(tmp_path: object) -> None:
    trace = _make_trace()
    save_trace(trace, str(tmp_path), golden=False)
    files = find_traces(str(tmp_path), "test_trace")
    with open(files[0]) as f:
        data = json.load(f)
    assert data["trace_id"] == "tid"
    assert data["format_version"] == "1"
