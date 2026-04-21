"""Deterministic terminal showcase for reagent-flow's Release Gatekeeper demo.

This script is optimized for product walkthroughs, GIFs, and first-time
evaluation. It exercises the same contract assertions as the LangGraph demo
without any live LLM calls, so the output is fast, stable, and easy to record.

Run with:
    cd examples/langgraph_demo
    uv run python showcase.py
"""

from __future__ import annotations

import shutil
import tempfile

import reagent_flow
from tools import RELEASES, _drift_release

GREEN = "\033[32m"
RED = "\033[31m"
ORANGE = "\033[38;5;208m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

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


def _divider(title: str) -> None:
    width = 72
    print(f"\n{ORANGE}{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}{RESET}\n")


def _seed_gatherer(
    trace_dir: str,
    version: str,
    *,
    drifted: bool,
) -> tuple[reagent_flow.Session, dict]:
    payload = _drift_release(RELEASES[version]) if drifted else RELEASES[version]
    with reagent_flow.session("gatekeeper-gatherer", trace_dir=trace_dir) as session:
        session.log_llm_call(
            tool_calls=[{"name": "get_release_info", "arguments": {"version": version}}],
        )
        session.log_tool_result("get_release_info", result=payload)
        session.log_llm_call(response_text="Release info gathered.", tool_calls=[])
    return session, payload


def _seed_assessor(
    trace_dir: str,
    *,
    parent: reagent_flow.Session,
    handoff: dict,
    risk_level: str,
) -> reagent_flow.Session:
    with reagent_flow.session(
        "gatekeeper-assessor",
        trace_dir=trace_dir,
        parent_trace_id=parent.trace.trace_id,
        handoff_context=handoff,
    ) as session:
        session.log_llm_call(
            tool_calls=[
                {
                    "name": "assess_risk",
                    "arguments": {
                        "release_version": handoff.get("release_version", "unknown"),
                        "risk_level": risk_level,
                        "justification": f"Deterministic {risk_level} for showcase.",
                    },
                }
            ],
        )
        session.log_tool_result(
            "assess_risk",
            result={
                "release_version": handoff.get("release_version", "unknown"),
                "risk_level": risk_level,
                "justification": f"Deterministic {risk_level} for showcase.",
            },
        )
        session.log_llm_call(response_text="Risk assessed.", tool_calls=[])
    return session


def main() -> None:
    """Run the deterministic broken-handoff showcase."""
    trace_dir = tempfile.mkdtemp()
    try:
        _divider("reagent-flow Showcase: Broken Handoff Caught Before Release")
        print(
            f"{DIM}Pipeline:{RESET} Gatherer -> Assessor -> Decider\n"
            f"{DIM}Focus:{RESET} upstream schema drift at the handoff boundary\n"
        )

        print(f"{BOLD}1. Stable contract{RESET}")
        print("   Assessor expects:")
        print("   - issues.open_p0: int")
        print("   - issues.open_p1: int\n")

        print(f"{BOLD}2. Upstream drift{RESET}")
        print("   Gatherer now returns:")
        print("   - issues.p0_count")
        print("   - open_p1 removed\n")

        gatherer, drifted_info = _seed_gatherer(trace_dir, "v2.3.1", drifted=True)
        assessor = _seed_assessor(
            trace_dir,
            parent=gatherer,
            handoff=drifted_info,
            risk_level="HIGH",
        )

        print(f"{BOLD}3. Contract assertion{RESET}")
        try:
            assessor.assert_handoff_matches(schema=RELEASE_INFO_SCHEMA)
        except AssertionError as exc:
            print(f"{RED}FAILED test_release_gatekeeper.py::test_gatherer_handoff{RESET}")
            print(str(exc))
            print(f"\n{GREEN}Result:{RESET} PR blocked before production.")
            return

        print("Unexpected: drifted handoff still matched the contract.")
    finally:
        shutil.rmtree(trace_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
