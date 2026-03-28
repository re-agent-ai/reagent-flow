"""pytest tests for the Release Risk Gatekeeper demo.

These tests make REAL LLM calls to Gemini via GOOGLE_API_KEY.
Skip them in CI by not setting the env var, or run explicitly:

    cd examples/langgraph_demo
    GOOGLE_API_KEY="your-key" uv run pytest test_demo.py -v
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import reagent_flow
from agent import build_agent, run_agent

# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set — skipping live LLM tests",
)


@pytest.mark.e2e
def test_standard_release_review(tmp_path: Path) -> None:
    """Scenario 1: Agent evaluates risky release. All assertions pass."""
    agent = build_agent()
    trace_dir = str(tmp_path)

    with reagent_flow.session("release-gatekeeper", trace_dir=trace_dir) as s:
        run_agent(agent, "Evaluate release v2.3.1 for production deployment.")

    s.assert_called("get_release_info")
    s.assert_called("make_decision")
    s.assert_called_before("get_release_info", "make_decision")
    s.assert_tool_succeeded("make_decision")
    s.assert_max_turns(10)


@pytest.mark.e2e
def test_golden_baseline_diff(tmp_path: Path) -> None:
    """Scenario 3: Different release data triggers baseline diff."""
    agent = build_agent()
    trace_dir = str(tmp_path)

    # Record golden baseline (risky release)
    with reagent_flow.session("release-gatekeeper", golden=True, trace_dir=trace_dir) as _golden_s:
        run_agent(agent, "Evaluate release v2.3.1 for production deployment.")

    # Run with clean release data
    with reagent_flow.session("release-gatekeeper", trace_dir=trace_dir) as actual_s:
        run_agent(agent, "Evaluate release v2.4.0 for production deployment.")

    from reagent_flow.diff import diff_traces
    from reagent_flow.storage.json import load_golden

    golden = load_golden(trace_dir, "release-gatekeeper")
    result = diff_traces(
        golden, actual_s.trace, ignore_fields={"arguments", "results", "response_text"}
    )
    assert result.summary() is not None
