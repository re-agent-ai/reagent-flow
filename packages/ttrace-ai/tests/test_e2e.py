"""End-to-end: record -> save -> load -> assert -> golden -> diff."""

from __future__ import annotations

from pathlib import Path

import pytest
import ttrace_ai


def _run_agent(session: ttrace_ai.Session) -> None:
    session.log_llm_call(
        tool_calls=[{"name": "search", "arguments": {"q": "test"}}],
        model="gpt-4o",
    )
    session.log_tool_result("search", result={"hits": 3}, duration_ms=150)
    session.log_llm_call(
        tool_calls=[{"name": "summarize", "arguments": {"hits": 3}}],
    )
    session.log_tool_result("summarize", result="3 results found", duration_ms=80)
    session.log_llm_call(response_text="Found 3 results.", tool_calls=[])


def test_record_and_assert(tmp_path: Path) -> None:
    with ttrace_ai.session("e2e_test", trace_dir=str(tmp_path)) as s:
        _run_agent(s)

    s.assert_called("search")
    s.assert_called("summarize")
    s.assert_called_before("search", "summarize")
    s.assert_tool_succeeded("search")
    s.assert_tool_succeeded("summarize")
    s.assert_never_called("delete")
    s.assert_max_turns(5)


def test_golden_baseline_match(tmp_path: Path) -> None:
    # Record golden
    with ttrace_ai.session("e2e_golden", golden=True, trace_dir=str(tmp_path)) as s:
        _run_agent(s)

    # Record actual — same behavior
    with ttrace_ai.session("e2e_golden", trace_dir=str(tmp_path)) as s2:
        _run_agent(s2)

    s2.assert_matches_baseline()


def test_golden_baseline_divergence(tmp_path: Path) -> None:
    # Record golden
    with ttrace_ai.session("e2e_div", golden=True, trace_dir=str(tmp_path)) as s:
        s.log_llm_call(tool_calls=[{"name": "lookup", "arguments": {"id": "1"}}])
        s.log_tool_result("lookup", result={"eligible": True})
        s.log_llm_call(
            tool_calls=[{"name": "process", "arguments": {"action": "approve"}}],
        )
        s.log_tool_result("process", result={"ok": True})

    # Record actual — different tool sequence
    with ttrace_ai.session("e2e_div", trace_dir=str(tmp_path)) as s2:
        s2.log_llm_call(tool_calls=[{"name": "lookup", "arguments": {"id": "1"}}])
        s2.log_tool_result("lookup", result={"eligible": False})
        s2.log_llm_call(tool_calls=[{"name": "reject", "arguments": {}}])
        s2.log_tool_result("reject", result={"ok": True})

    with pytest.raises(AssertionError, match="does not match golden baseline"):
        s2.assert_matches_baseline()


def test_trace_file_persisted(tmp_path: Path) -> None:
    with ttrace_ai.session("persist_test", trace_dir=str(tmp_path)) as s:
        s.log_llm_call(response_text="hello", tool_calls=[])

    traces_dir = tmp_path / "traces"
    assert traces_dir.exists()
    files = list(traces_dir.glob("persist_test_*.trace.json"))
    assert len(files) == 1
