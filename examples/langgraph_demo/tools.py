"""Release Risk Gatekeeper tools — one tool per sub-agent.

Three LangGraph ReAct agents cooperate to evaluate a release:

    Gatherer (get_release_info)
        -> handoff -> Assessor (assess_risk)
                          -> handoff -> Decider (make_decision)

Each agent owns exactly one tool. The orchestrator in ``orchestrator.py``
extracts structured output from each stage and passes it to the next as
``handoff_context`` so reagent-flow can validate the contract between
agents.

``get_release_info_drifted`` is a deliberately broken variant used by the
"caught before release" demo scenario — it renames ``issues.open_p0`` and
drops ``issues.open_p1`` to simulate an upstream refactor gone wrong.
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Release data fixtures — simulate real CI/CD system responses
# ---------------------------------------------------------------------------

RELEASES: dict[str, dict] = {
    "v2.3.1": {
        "release_version": "v2.3.1",
        "branch": "release/2.3.1",
        "ci": {
            "pipeline": "passed",
            "coverage": 87.3,
            "unit_failed": 0,
            "integration_failed": 2,
            "e2e_failed": 1,
        },
        "issues": {
            "open_p0": 1,
            "open_p1": 3,
            "resolved_this_release": 12,
        },
        "deploy_history": [
            {"version": "v2.3.0", "date": "2026-03-24", "status": "stable", "rollback": False},
            {"version": "v2.2.9", "date": "2026-03-18", "status": "rolled_back", "rollback": True},
        ],
    },
    "v2.4.0": {
        "release_version": "v2.4.0",
        "branch": "release/2.4.0",
        "ci": {
            "pipeline": "passed",
            "coverage": 91.2,
            "unit_failed": 0,
            "integration_failed": 0,
            "e2e_failed": 0,
        },
        "issues": {
            "open_p0": 0,
            "open_p1": 1,
            "resolved_this_release": 18,
        },
        "deploy_history": [
            {"version": "v2.3.2", "date": "2026-03-26", "status": "stable", "rollback": False},
            {"version": "v2.3.1", "date": "2026-03-25", "status": "stable", "rollback": False},
        ],
    },
}


# ---------------------------------------------------------------------------
# Stable tools — one per sub-agent
# ---------------------------------------------------------------------------


@tool
def get_release_info(version: str) -> str:
    """Look up release information from CI/CD systems.

    Returns a JSON object with branch, CI test results, open issue counts,
    and deployment history for the given release version.

    Args:
        version: The release version to look up (e.g., "v2.3.1").
    """
    data = RELEASES.get(version)
    if data is None:
        return json.dumps({"error": f"Release {version} not found"})
    return json.dumps(data)


@tool
def assess_risk(release_version: str, risk_level: str, justification: str) -> str:
    """Record a risk assessment for a release.

    The LLM reasons about the release data passed to it and fills in the
    structured fields. The tool echoes the assessment so reagent-flow can
    capture it from the trace and hand it off to the decision agent.

    Args:
        release_version: The version being assessed.
        risk_level: One of "LOW", "MEDIUM", or "HIGH".
        justification: A short explanation for the assigned risk level.
    """
    return json.dumps(
        {
            "release_version": release_version,
            "risk_level": risk_level,
            "justification": justification,
        }
    )


@tool
def make_decision(release_version: str, decision: str, reason: str) -> str:
    """Record the final deployment decision.

    The LLM uses the upstream risk assessment to choose APPROVE or BLOCK
    and provides a short reason. The tool echoes the decision so it lands
    in the trace as a distinct step.

    Args:
        release_version: The version being decided on.
        decision: Either "APPROVE" or "BLOCK".
        reason: A short explanation for the decision.
    """
    return json.dumps(
        {
            "release_version": release_version,
            "decision": decision,
            "reason": reason,
        }
    )


# ---------------------------------------------------------------------------
# Drifted gatherer — simulates an upstream refactor that breaks the contract
# ---------------------------------------------------------------------------


def _drift_release(data: dict) -> dict:
    """Return a copy of release data with the ``issues`` block refactored.

    Simulates a real-world drift: the upstream team renamed ``open_p0`` to
    ``p0_count`` and dropped ``open_p1`` entirely. Downstream agents may
    keep running — the LLM is lenient — but the handoff contract will catch
    it.
    """
    drifted = {k: v for k, v in data.items() if k != "issues"}
    drifted["issues"] = {
        "p0_count": data["issues"]["open_p0"],
        "resolved_this_release": data["issues"]["resolved_this_release"],
    }
    return drifted


@tool("get_release_info")
def get_release_info_drifted(version: str) -> str:
    """Look up release information (drifted schema — simulated regression).

    Same purpose and name as ``get_release_info`` — the upstream team
    shipped a refactor of the same tool — but this variant's ``issues``
    block has been renamed: ``open_p0`` is now ``p0_count`` and
    ``open_p1`` has been removed. Used by the demo's "broken handoff"
    scenario to show reagent-flow catching upstream drift exactly the
    way it happens in the real world (same tool name, different shape).

    Args:
        version: The release version to look up (e.g., "v2.3.1").
    """
    data = RELEASES.get(version)
    if data is None:
        return json.dumps({"error": f"Release {version} not found"})
    return json.dumps(_drift_release(data))
