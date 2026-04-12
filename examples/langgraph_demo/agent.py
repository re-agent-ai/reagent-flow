"""Sub-agent builders and runner for the Release Risk Gatekeeper demo.

Three small LangGraph ReAct agents cooperate via handoffs:

    Gatherer  — owns ``get_release_info``, loads raw release data.
    Assessor  — owns ``assess_risk``, reasons about the data from the gatherer.
    Decider   — owns ``make_decision``, issues APPROVE or BLOCK.

Each agent gets only the single tool it needs so the trace for each phase
is focused and the handoff boundary between phases is explicit.
"""

from __future__ import annotations

from typing import Any

GATHERER_PROMPT = (
    "You are the Release Gatherer. Your only job is to call "
    "get_release_info with the version you are given. Do not editorialize, "
    "do not skip the tool call, and do not invent data. After the tool "
    "returns, reply with a one-line acknowledgement."
)

ASSESSOR_PROMPT = (
    "You are the Release Risk Assessor. You will be given release data as "
    "a JSON object. Your job is to call assess_risk exactly once with:\n"
    "  - release_version: the version string from the data\n"
    '  - risk_level: "LOW", "MEDIUM", or "HIGH" based on failing tests, '
    "open P0/P1 issues, and rollback history\n"
    "  - justification: one sentence explaining your reasoning\n"
    "Any release with an open P0 or a recent rollback is HIGH risk."
)

DECIDER_PROMPT = (
    "You are the Release Decider. You will be given a risk assessment as a "
    "JSON object. Your job is to call make_decision exactly once with:\n"
    "  - release_version: the version from the assessment\n"
    '  - decision: "APPROVE" or "BLOCK"\n'
    "  - reason: one sentence explaining the call\n"
    "BLOCK any release whose risk_level is HIGH. APPROVE LOW risk releases. "
    "For MEDIUM risk, use your judgement."
)


def _build_agent(tool: Any, prompt: str) -> Any:
    """Build a single-tool LangGraph ReAct agent."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langgraph.prebuilt import create_react_agent

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    return create_react_agent(llm, [tool], prompt=prompt)


def build_gatherer_agent(*, drifted: bool = False) -> Any:
    """Build the gatherer sub-agent.

    Args:
        drifted: If True, use the ``get_release_info_drifted`` tool whose
            output breaks the downstream handoff contract. Used by the
            "broken handoff caught" scenario.
    """
    from tools import get_release_info, get_release_info_drifted

    tool = get_release_info_drifted if drifted else get_release_info
    return _build_agent(tool, GATHERER_PROMPT)


def build_assessor_agent() -> Any:
    """Build the assessor sub-agent."""
    from tools import assess_risk

    return _build_agent(assess_risk, ASSESSOR_PROMPT)


def build_decider_agent() -> Any:
    """Build the decider sub-agent."""
    from tools import make_decision

    return _build_agent(make_decision, DECIDER_PROMPT)


def run_agent(agent: Any, task: str) -> Any:
    """Run an agent with reagent-flow tracing via ReagentGraphTracer callback."""
    from reagent_flow_langgraph import ReagentGraphTracer

    tracer = ReagentGraphTracer()
    return agent.invoke(
        {"messages": [("user", task)]},
        config={"callbacks": [tracer]},
    )
