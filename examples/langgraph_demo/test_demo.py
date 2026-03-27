"""pytest tests for the Release Risk Gatekeeper demo.

These tests make REAL LLM calls to Gemini via GOOGLE_API_KEY.
Skip them in CI by not setting the env var, or run explicitly:

    cd examples/langgraph_demo
    GOOGLE_API_KEY="your-key" uv run pytest test_demo.py -v
"""

from __future__ import annotations

import os
import tempfile

import pytest

import reagent_flow

# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set — skipping live LLM tests",
)


def _build_agent(system_prompt: str | None = None):
    """Build the gatekeeper agent."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langgraph.prebuilt import create_react_agent

    from tools import assess_risk, get_release_info, make_decision

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    tools = [get_release_info, assess_risk, make_decision]

    prompt = system_prompt or (
        "You are a Release Risk Gatekeeper. Your job is to evaluate whether "
        "a software release is safe to deploy to production.\n\n"
        "For EVERY release review, you MUST follow this exact process:\n"
        "1. Call get_release_info to gather release data\n"
        "2. Call assess_risk with the release data to produce a risk assessment\n"
        "3. Call make_decision with the risk assessment to make a final APPROVE/BLOCK decision\n\n"
        "Never skip a step. Never make a decision without assessing risk first."
    )

    return create_react_agent(llm, tools, prompt=prompt)


def _run_agent(agent, task: str, session: reagent_flow.Session):
    """Run the agent with reagent-flow tracing."""
    from reagent_flow_langgraph import ReagentGraphTracer

    tracer = ReagentGraphTracer()
    return agent.invoke(
        {"messages": [("user", task)]},
        config={"callbacks": [tracer]},
    )


@pytest.mark.e2e
def test_standard_release_review() -> None:
    """Scenario 1: Agent evaluates risky release. All assertions pass."""
    agent = _build_agent()
    trace_dir = tempfile.mkdtemp()

    with reagent_flow.session("release-gatekeeper", trace_dir=trace_dir) as s:
        _run_agent(agent, "Evaluate release v2.3.1 for production deployment.", s)

    s.assert_called("get_release_info")
    s.assert_called("make_decision")
    s.assert_called_before("get_release_info", "make_decision")
    s.assert_tool_succeeded("make_decision")
    s.assert_max_turns(10)


@pytest.mark.e2e
def test_golden_baseline_diff() -> None:
    """Scenario 3: Different release data triggers baseline diff."""
    agent = _build_agent()
    trace_dir = tempfile.mkdtemp()

    # Record golden baseline (risky release)
    with reagent_flow.session(
        "release-gatekeeper", golden=True, trace_dir=trace_dir
    ) as golden_s:
        _run_agent(
            agent, "Evaluate release v2.3.1 for production deployment.", golden_s
        )

    # Run with clean release data
    with reagent_flow.session("release-gatekeeper", trace_dir=trace_dir) as actual_s:
        _run_agent(
            agent, "Evaluate release v2.4.0 for production deployment.", actual_s
        )

    # Ignore arguments and results (LLM output is non-deterministic)
    # but tool call sequence (names) should still be compared
    from reagent_flow.diff import diff_traces
    from reagent_flow.storage.json import load_golden

    golden = load_golden(trace_dir, "release-gatekeeper")
    result = diff_traces(
        golden, actual_s.trace, ignore_fields={"arguments", "results", "response_text"}
    )
    # Just verify the diff engine runs without error
    assert result.summary() is not None
