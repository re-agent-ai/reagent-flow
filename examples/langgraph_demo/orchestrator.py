"""Multi-agent pipeline orchestrator for the Release Risk Gatekeeper demo.

Runs three sub-agent sessions in sequence and wires the output of each
phase into the ``handoff_context`` of the next, so reagent-flow's contract
assertions have something to validate against.

    gatherer -> release_info  -> assessor -> risk_assessment -> decider

Each phase owns its own ``reagent_flow.session``. Parent/child linking is
done via ``parent_trace_id`` on the session constructor. The extraction
helpers pull structured data out of the just-completed session's trace so
it can flow forward as the next session's handoff context.
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
from reagent_flow.models import Trace


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
# Pipeline
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
            schema breaks the downstream handoff contract. The pipeline
            still runs to completion; the contract assertion is what
            catches it.
    """
    # --- Phase 1: Gatherer ---
    gatherer = build_gatherer_agent(drifted=drifted_gatherer)
    with reagent_flow.session(
        "gatekeeper-gatherer",
        golden=golden,
        trace_dir=trace_dir,
    ) as s_gather:
        run_agent(gatherer, f"Look up release information for {version}.")
    release_info = _as_dict(_last_tool_result(s_gather.trace, "get_release_info"))

    # --- Phase 2: Assessor (child of gatherer) ---
    assessor = build_assessor_agent()
    with reagent_flow.session(
        "gatekeeper-assessor",
        golden=golden,
        trace_dir=trace_dir,
        parent_trace_id=s_gather.trace.trace_id,
        handoff_context=release_info,
    ) as s_assess:
        run_agent(
            assessor,
            "Assess the risk of this release based on the following data. "
            f"Call assess_risk exactly once.\n\nRelease data:\n{json.dumps(release_info)}",
        )
    risk_assessment = _as_dict(_last_tool_result(s_assess.trace, "assess_risk"))
    if not risk_assessment:
        # Fallback: if the tool echoed as non-dict, rebuild from the
        # LLM's call arguments which carry the same structured fields.
        risk_assessment = _last_tool_arguments(s_assess.trace, "assess_risk") or {}

    # --- Phase 3: Decider (child of assessor) ---
    decider = build_decider_agent()
    with reagent_flow.session(
        "gatekeeper-decider",
        golden=golden,
        trace_dir=trace_dir,
        parent_trace_id=s_assess.trace.trace_id,
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

    return PipelineResult(
        gatherer=s_gather,
        assessor=s_assess,
        decider=s_decide,
        release_info=release_info,
        risk_assessment=risk_assessment,
        decision=decision,
    )
