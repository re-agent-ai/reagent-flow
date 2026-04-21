"""Multi-agent pipeline orchestrator for the Release Risk Gatekeeper demo.

Runs three sub-agent sessions as nodes in a LangGraph StateGraph. Each node
opens its own reagent-flow session, runs the agent, extracts the structured
output from the trace, and passes it forward via graph state.

    gatherer -> release_info  -> assessor -> risk_assessment -> decider

Parent/child linking is done via ``parent_trace_id`` on the session
constructor. The extraction helpers pull structured data out of each
session's trace so it can flow forward as the next node's handoff context.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import reagent_flow
from agent import (
    build_assessor_agent,
    build_decider_agent,
    build_gatherer_agent,
    run_agent,
)
from langgraph.graph import END, START, StateGraph  # type: ignore[import-untyped]
from reagent_flow.models import Trace
from typing_extensions import TypedDict


@dataclass
class PipelineResult:
    """All the sessions and payloads produced by one pipeline run."""

    gatherer: reagent_flow.Session
    assessor: reagent_flow.Session
    decider: reagent_flow.Session
    release_info: dict[str, Any]
    risk_assessment: dict[str, Any]
    decision: dict[str, Any]


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class PipelineState(TypedDict):
    """State flowing through the LangGraph pipeline."""

    # Input config — set once at invocation, read by nodes
    version: str
    trace_dir: str
    golden: bool
    drifted_gatherer: bool

    # Phase outputs — populated by each node
    release_info: dict[str, Any]
    risk_assessment: dict[str, Any]
    decision: dict[str, Any]

    # Session objects — stored for post-run assertions
    gatherer_session: Any
    assessor_session: Any
    decider_session: Any


# ---------------------------------------------------------------------------
# Trace extraction helpers
# ---------------------------------------------------------------------------


def _last_tool_result(trace: Trace, tool_name: str) -> Any:
    """Return the final non-error result recorded for ``tool_name``."""
    latest: Any = None
    for turn in trace.turns:
        call_ids = {tc.call_id for tc in turn.llm_call.tool_calls if tc.name == tool_name}
        for tr in turn.tool_results:
            if tr.call_id in call_ids and tr.error is None:
                latest = tr.result
    return latest


def _last_tool_arguments(trace: Trace, tool_name: str) -> dict[str, Any] | None:
    """Return the arguments of the final recorded call to ``tool_name``."""
    latest: dict[str, Any] | None = None
    for turn in trace.turns:
        for tc in turn.llm_call.tool_calls:
            if tc.name == tool_name:
                latest = dict(tc.arguments)
    return latest


def _as_dict(value: Any) -> dict[str, Any]:
    """Coerce a tool result (possibly a JSON string) to a dict."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


def gatherer_node(state: PipelineState) -> dict[str, Any]:
    """Phase 1: Gather release data."""
    gatherer = build_gatherer_agent(drifted=state["drifted_gatherer"])
    with reagent_flow.session(
        "gatekeeper-gatherer",
        golden=state["golden"],
        trace_dir=state["trace_dir"],
    ) as s_gather:
        run_agent(gatherer, f"Look up release information for {state['version']}.")
    release_info = _as_dict(_last_tool_result(s_gather.trace, "get_release_info"))
    return {
        "gatherer_session": s_gather,
        "release_info": release_info,
    }


def assessor_node(state: PipelineState) -> dict[str, Any]:
    """Phase 2: Assess release risk based on gathered data."""
    assessor = build_assessor_agent()
    release_info = state["release_info"]
    with reagent_flow.session(
        "gatekeeper-assessor",
        golden=state["golden"],
        trace_dir=state["trace_dir"],
        parent_trace_id=state["gatherer_session"].trace.trace_id,
        handoff_context=release_info,
    ) as s_assess:
        run_agent(
            assessor,
            "Assess the risk of this release based on the following data. "
            f"Call assess_risk exactly once.\n\nRelease data:\n{json.dumps(release_info)}",
        )
    risk_assessment = _as_dict(_last_tool_result(s_assess.trace, "assess_risk"))
    if not risk_assessment:
        risk_assessment = _last_tool_arguments(s_assess.trace, "assess_risk") or {}
    return {
        "assessor_session": s_assess,
        "risk_assessment": risk_assessment,
    }


def decider_node(state: PipelineState) -> dict[str, Any]:
    """Phase 3: Make the final deployment decision."""
    decider = build_decider_agent()
    risk_assessment = state["risk_assessment"]
    with reagent_flow.session(
        "gatekeeper-decider",
        golden=state["golden"],
        trace_dir=state["trace_dir"],
        parent_trace_id=state["assessor_session"].trace.trace_id,
        handoff_context=risk_assessment,
    ) as s_decide:
        run_agent(
            decider,
            "Make the final deployment decision based on this risk assessment. "
            "Call make_decision exactly once.\n\nRisk assessment:\n"
            f"{json.dumps(risk_assessment)}",
        )
    decision = _as_dict(_last_tool_result(s_decide.trace, "make_decision"))
    if not decision:
        decision = _last_tool_arguments(s_decide.trace, "make_decision") or {}
    return {
        "decider_session": s_decide,
        "decision": decision,
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def _build_graph() -> Any:
    """Build and compile the three-phase pipeline graph."""
    graph = StateGraph(PipelineState)
    graph.add_node("gatherer", gatherer_node)
    graph.add_node("assessor", assessor_node)
    graph.add_node("decider", decider_node)
    graph.add_edge(START, "gatherer")
    graph.add_edge("gatherer", "assessor")
    graph.add_edge("assessor", "decider")
    graph.add_edge("decider", END)
    return graph.compile()


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def run_pipeline(
    version: str,
    *,
    trace_dir: str,
    golden: bool = False,
    drifted_gatherer: bool = False,
) -> PipelineResult:
    """Run the three-phase release gatekeeper pipeline for a release version.

    Args:
        version: Release version to evaluate (e.g. ``"v2.3.1"``).
        trace_dir: Directory where reagent-flow writes trace JSON.
        golden: If True, save each session's trace as a golden baseline.
        drifted_gatherer: If True, use the drifted gatherer whose output
            schema breaks the downstream handoff contract.
    """
    pipeline = _build_graph()
    final_state = pipeline.invoke(
        {
            "version": version,
            "trace_dir": trace_dir,
            "golden": golden,
            "drifted_gatherer": drifted_gatherer,
            "release_info": {},
            "risk_assessment": {},
            "decision": {},
            "gatherer_session": None,
            "assessor_session": None,
            "decider_session": None,
        }
    )
    return PipelineResult(
        gatherer=final_state["gatherer_session"],
        assessor=final_state["assessor_session"],
        decider=final_state["decider_session"],
        release_info=final_state["release_info"],
        risk_assessment=final_state["risk_assessment"],
        decision=final_state["decision"],
    )
