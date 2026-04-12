"""pytest tests for the Release Risk Gatekeeper demo's contract story.

These tests exercise the contract-assertion API the demo showcases without
making any live LLM calls. They construct reagent-flow sessions by hand
using ``log_llm_call`` / ``log_tool_result`` so they can run in CI with no
API key (per the repo rule: never make real LLM API calls in tests).

For the live end-to-end runnable version that actually drives Gemini
through the multi-agent pipeline, see ``demo.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import reagent_flow
from tools import RELEASES, _drift_release

# ---------------------------------------------------------------------------
# Schemas — mirror the ones used in demo.py
# ---------------------------------------------------------------------------

RELEASE_INFO_SCHEMA = {
    "release_version": str,
    "branch": str,
    "ci": {
        "pipeline": str,
        "coverage": float,
        "unit_failed": int,
        "integration_failed": int,
        "e2e_failed": int,
    },
    "issues": {
        "open_p0": int,
        "open_p1": int,
        "resolved_this_release": int,
    },
    "deploy_history": [
        {"version": str, "date": str, "status": str, "rollback": bool},
    ],
}

RISK_ASSESSMENT_SCHEMA = {
    "release_version": str,
    "risk_level": str,
    "justification": str,
}

DECISION_SCHEMA = {"release_version": str, "decision": str, "reason": str}


# ---------------------------------------------------------------------------
# Manual pipeline — mirrors orchestrator.run_pipeline() without calling an LLM
# ---------------------------------------------------------------------------


def _seed_gatherer(
    trace_dir: str, version: str, *, drifted: bool, golden: bool = False
) -> tuple[reagent_flow.Session, dict]:
    """Build a gatherer session and log a single get_release_info call.

    Returns the closed session and the tool's parsed output dict, ready to
    be used as the assessor's ``handoff_context``.
    """
    data = RELEASES[version]
    payload = _drift_release(data) if drifted else data
    with reagent_flow.session("gatekeeper-gatherer", golden=golden, trace_dir=trace_dir) as s:
        s.log_llm_call(
            tool_calls=[{"name": "get_release_info", "arguments": {"version": version}}],
        )
        s.log_tool_result("get_release_info", result=payload)
        s.log_llm_call(response_text="Release info gathered.", tool_calls=[])
    return s, payload


def _seed_assessor(
    trace_dir: str,
    *,
    parent: reagent_flow.Session,
    handoff: dict,
    risk_level: str,
    golden: bool = False,
) -> tuple[reagent_flow.Session, dict]:
    version = handoff.get("release_version", "unknown")
    assessment = {
        "release_version": version,
        "risk_level": risk_level,
        "justification": f"Deterministic {risk_level} for test fixture.",
    }
    with reagent_flow.session(
        "gatekeeper-assessor",
        golden=golden,
        trace_dir=trace_dir,
        parent_trace_id=parent.trace.trace_id,
        handoff_context=handoff,
    ) as s:
        s.log_llm_call(
            tool_calls=[{"name": "assess_risk", "arguments": assessment}],
        )
        s.log_tool_result("assess_risk", result=assessment)
        s.log_llm_call(response_text="Risk assessed.", tool_calls=[])
    return s, assessment


def _seed_decider(
    trace_dir: str,
    *,
    parent: reagent_flow.Session,
    handoff: dict,
    decision: str,
    golden: bool = False,
) -> tuple[reagent_flow.Session, dict]:
    version = handoff.get("release_version", "unknown")
    outcome = {
        "release_version": version,
        "decision": decision,
        "reason": f"Deterministic {decision} for test fixture.",
    }
    with reagent_flow.session(
        "gatekeeper-decider",
        golden=golden,
        trace_dir=trace_dir,
        parent_trace_id=parent.trace.trace_id,
        handoff_context=handoff,
    ) as s:
        s.log_llm_call(
            tool_calls=[{"name": "make_decision", "arguments": outcome}],
        )
        s.log_tool_result("make_decision", result=outcome)
        s.log_llm_call(response_text="Decision made.", tool_calls=[])
    return s, outcome


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_green_path_contracts_hold(tmp_path: Path) -> None:
    """Every handoff + tool-output contract holds on a clean pipeline."""
    trace_dir = str(tmp_path)

    gatherer, release_info = _seed_gatherer(trace_dir, "v2.3.1", drifted=False)
    assessor, _ = _seed_assessor(
        trace_dir, parent=gatherer, handoff=release_info, risk_level="HIGH"
    )
    decider, _ = _seed_decider(
        trace_dir,
        parent=assessor,
        handoff={
            "release_version": "v2.3.1",
            "risk_level": "HIGH",
            "justification": "seed",
        },
        decision="BLOCK",
    )

    # Tool output contract on the gatherer
    gatherer.assert_tool_output_matches("get_release_info", schema=RELEASE_INFO_SCHEMA)

    # Parent/child linking and handoff contract at gatherer -> assessor
    assessor.assert_handoff_received(gatherer)
    assessor.assert_handoff_matches(schema=RELEASE_INFO_SCHEMA)
    assessor.assert_called_times("assess_risk", min=1, max=1)

    # Parent/child linking and handoff contract at assessor -> decider
    decider.assert_handoff_received(assessor)
    decider.assert_handoff_matches(schema=RISK_ASSESSMENT_SCHEMA)
    decider.assert_tool_output_matches("make_decision", schema=DECISION_SCHEMA)


def test_broken_handoff_is_caught(tmp_path: Path) -> None:
    """Upstream drift in the gatherer fails the handoff contract.

    Simulates the real-world scenario: the gatherer tool keeps its name
    but its payload has refactored ``issues.open_p0`` to ``p0_count`` and
    dropped ``issues.open_p1``. The assessor session receives the drifted
    dict as its handoff_context, and ``assert_handoff_matches`` rejects it
    with the exact missing field path.
    """
    trace_dir = str(tmp_path)

    gatherer, drifted_info = _seed_gatherer(trace_dir, "v2.3.1", drifted=True)
    assessor, _ = _seed_assessor(
        trace_dir, parent=gatherer, handoff=drifted_info, risk_level="HIGH"
    )

    try:
        assessor.assert_handoff_matches(schema=RELEASE_INFO_SCHEMA)
    except AssertionError as exc:
        message = str(exc)
        assert "issues.open_p0" in message, (
            f"expected drift error to name issues.open_p0, got:\n{message}"
        )
        assert "missing" in message
        return

    raise AssertionError(
        "Expected assert_handoff_matches to fail on the drifted payload, "
        "but it passed. Drifted payload was: " + json.dumps(drifted_info)
    )


def test_context_preserved_across_two_hops(tmp_path: Path) -> None:
    """The version identifier flows unchanged through both handoffs."""
    trace_dir = str(tmp_path)

    gatherer, release_info = _seed_gatherer(trace_dir, "v2.3.1", drifted=False)
    assessor, assessment = _seed_assessor(
        trace_dir, parent=gatherer, handoff=release_info, risk_level="HIGH"
    )
    decider, _ = _seed_decider(trace_dir, parent=assessor, handoff=assessment, decision="BLOCK")

    assessor.assert_context_preserved({"release_version": "v2.3.1"}, fields=["release_version"])
    decider.assert_context_preserved({"release_version": "v2.3.1"}, fields=["release_version"])


def test_drifted_tool_still_registered_as_get_release_info() -> None:
    """The drifted tool shares the stable tool's name.

    This keeps the trace extraction in ``orchestrator.py`` correct without
    having to know which implementation was wired in, and mirrors how real
    upstream drift surfaces (same tool name, different shape).
    """
    from tools import get_release_info, get_release_info_drifted

    assert get_release_info.name == "get_release_info"
    assert get_release_info_drifted.name == "get_release_info"
