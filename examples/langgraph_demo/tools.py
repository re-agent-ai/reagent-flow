"""Release Risk Gatekeeper tools.

Three tools that a LangGraph ReAct agent uses to evaluate release safety.
`get_release_info` returns hardcoded but realistic CI/CD data.
`assess_risk` and `make_decision` are pass-through tools whose output
is determined by the LLM's reasoning.
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Release data fixtures — simulate real CI/CD system responses
# ---------------------------------------------------------------------------

RELEASES: dict[str, dict] = {
    "v2.3.1": {
        "version": "v2.3.1",
        "git": {
            "branch": "release/2.3.1",
            "commits_since_last_release": 14,
            "authors": ["alice", "bob", "charlie"],
        },
        "ci": {
            "pipeline": "passed",
            "test_suites": {
                "unit": {"passed": 312, "failed": 0, "skipped": 4},
                "integration": {"passed": 87, "failed": 2, "skipped": 0},
                "e2e": {"passed": 45, "failed": 1, "skipped": 2},
            },
            "coverage": 87.3,
            "build_duration_sec": 342,
        },
        "issues": {
            "open_p0": 1,
            "open_p1": 3,
            "resolved_this_release": 12,
        },
        "deploy_history": [
            {
                "version": "v2.3.0",
                "date": "2026-03-24",
                "status": "stable",
                "rollback": False,
            },
            {
                "version": "v2.2.9",
                "date": "2026-03-18",
                "status": "rolled_back",
                "rollback": True,
            },
        ],
    },
    "v2.3.2": {
        "version": "v2.3.2",
        "git": {
            "branch": "hotfix/2.3.2",
            "commits_since_last_release": 2,
            "authors": ["alice"],
        },
        "ci": {
            "pipeline": "passed",
            "test_suites": {
                "unit": {"passed": 316, "failed": 0, "skipped": 4},
                "integration": {"passed": 89, "failed": 0, "skipped": 0},
                "e2e": {"passed": 48, "failed": 0, "skipped": 0},
            },
            "coverage": 88.1,
            "build_duration_sec": 298,
        },
        "issues": {
            "open_p0": 0,
            "open_p1": 2,
            "resolved_this_release": 1,
        },
        "deploy_history": [
            {
                "version": "v2.3.1",
                "date": "2026-03-25",
                "status": "stable",
                "rollback": False,
            },
        ],
    },
    "v2.4.0": {
        "version": "v2.4.0",
        "git": {
            "branch": "release/2.4.0",
            "commits_since_last_release": 31,
            "authors": ["alice", "bob", "charlie", "diana"],
        },
        "ci": {
            "pipeline": "passed",
            "test_suites": {
                "unit": {"passed": 340, "failed": 0, "skipped": 3},
                "integration": {"passed": 92, "failed": 0, "skipped": 0},
                "e2e": {"passed": 51, "failed": 0, "skipped": 1},
            },
            "coverage": 91.2,
            "build_duration_sec": 310,
        },
        "issues": {
            "open_p0": 0,
            "open_p1": 1,
            "resolved_this_release": 18,
        },
        "deploy_history": [
            {
                "version": "v2.3.2",
                "date": "2026-03-26",
                "status": "stable",
                "rollback": False,
            },
            {
                "version": "v2.3.1",
                "date": "2026-03-25",
                "status": "stable",
                "rollback": False,
            },
        ],
    },
}


@tool
def get_release_info(version: str) -> str:
    """Look up release information from CI/CD systems.

    Returns test results, coverage, open issues, and deployment history
    for the given release version. Use this tool FIRST to gather data
    before assessing risk.

    Args:
        version: The release version to look up (e.g., "v2.3.1").
    """
    data = RELEASES.get(version)
    if data is None:
        return json.dumps({"error": f"Release {version} not found"})
    return json.dumps(data, indent=2)


@tool
def assess_risk(release_info: str) -> str:
    """Assess the risk level of a release based on its information.

    Tracing checkpoint: the LLM produces a risk assessment (LOW/MEDIUM/HIGH
    with justification) as the argument. The tool returns it unchanged so
    reagent-flow records it as a distinct step in the trace.

    Args:
        release_info: The LLM's risk analysis of the release data.
    """
    return release_info


@tool
def make_decision(risk_assessment: str) -> str:
    """Make a final deployment decision: APPROVE or BLOCK.

    Tracing checkpoint: the LLM produces an APPROVE/BLOCK decision as the
    argument. The tool returns it unchanged so reagent-flow records it as
    a distinct step in the trace.

    Args:
        risk_assessment: The LLM's deployment decision with justification.
    """
    return risk_assessment
