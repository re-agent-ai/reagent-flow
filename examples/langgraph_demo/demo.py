"""
reagent-flow + LangGraph demo: Release Risk Gatekeeper (multi-agent handoffs)

Three sub-agents cooperate to evaluate a release:

    Gatherer -> Assessor -> Decider

Each phase runs in its own reagent-flow session linked via parent_trace_id
and handoff_context. reagent-flow's contract assertions validate the shape
of the data flowing between agents — which is exactly where real
multi-agent systems break.

Run with:
    cd examples/langgraph_demo
    GOOGLE_API_KEY="your-key" uv run python demo.py

Three scenarios:
  1. Green path         — all handoff contracts hold, pipeline approves.
  2. Broken handoff     — an upstream refactor drops a field; reagent-flow
                          catches it at the gatherer -> assessor boundary.
  3. Regression diff    — clean release against the golden baseline shows
                          the behavioral diff engine in action.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

from orchestrator import run_pipeline

# ---------------------------------------------------------------------------
# Terminal colors
# ---------------------------------------------------------------------------

GREEN = "\033[32m"
RED = "\033[31m"
ORANGE = "\033[38;5;208m"
YELLOW = "\033[38;5;220m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def divider(title: str) -> None:
    """Print a section divider."""
    width = 70
    print(f"\n{ORANGE}{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}{RESET}\n")


# ---------------------------------------------------------------------------
# Schemas — declared once, reused across scenarios
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


# ---------------------------------------------------------------------------
# Scenario 1 — Green path: all contracts hold
# ---------------------------------------------------------------------------


def scenario_1_green_path(trace_dir: str) -> None:
    """Risky release v2.3.1 — pipeline runs end-to-end, all contracts valid."""
    divider("SCENARIO 1: Multi-Agent Pipeline (Green Path)")

    print(f"{DIM}Task: Evaluate release v2.3.1 through Gatherer -> Assessor -> Decider{RESET}")
    print(f"{DIM}Expected: all handoff contracts pass, agent chain blocks a risky release{RESET}\n")

    result = run_pipeline("v2.3.1", trace_dir=trace_dir, golden=True)

    passed: list[str] = []
    try:
        # Gatherer owns the release info contract.
        result.gatherer.assert_tool_output_matches("get_release_info", schema=RELEASE_INFO_SCHEMA)
        passed.append("gatherer.assert_tool_output_matches('get_release_info', release_schema)")

        # Assessor session is a child of the gatherer session.
        result.assessor.assert_handoff_received(result.gatherer)
        passed.append("assessor.assert_handoff_received(gatherer)")

        # The handoff from gatherer -> assessor must match the contract.
        result.assessor.assert_handoff_matches(schema=RELEASE_INFO_SCHEMA)
        passed.append("assessor.assert_handoff_matches(release_schema)")

        # The assessor must have called its one tool exactly once.
        result.assessor.assert_called_times("assess_risk", min=1, max=1)
        passed.append("assessor.assert_called_times('assess_risk', min=1, max=1)")

        # Decider is a child of the assessor and receives a risk assessment.
        result.decider.assert_handoff_received(result.assessor)
        passed.append("decider.assert_handoff_received(assessor)")

        result.decider.assert_handoff_matches(schema=RISK_ASSESSMENT_SCHEMA)
        passed.append("decider.assert_handoff_matches(risk_assessment_schema)")

        # Version must survive both hops unchanged.
        result.decider.assert_context_preserved(
            {"release_version": "v2.3.1"}, fields=["release_version"]
        )
        passed.append("decider.assert_context_preserved({'release_version': 'v2.3.1'})")

        # Decision contract on the last tool.
        result.decider.assert_tool_output_matches(
            "make_decision",
            schema={"release_version": str, "decision": str, "reason": str},
        )
        passed.append("decider.assert_tool_output_matches('make_decision', decision_schema)")

    except AssertionError as e:
        print(f"{RED}Unexpected assertion failure in green path:{RESET}")
        print(str(e))
        return

    print(f"{GREEN}All contract assertions passed:{RESET}")
    for a in passed:
        print(f"  {GREEN}+{RESET} {a}")
    decision = result.decision.get("decision", "?")
    print(f"\n{DIM}Final decision: {decision} — {result.decision.get('reason', '')}{RESET}")
    print(f"{DIM}Golden baseline saved to {trace_dir}/golden/{RESET}")


# ---------------------------------------------------------------------------
# Scenario 2 — Broken handoff: upstream drift caught at the boundary
# ---------------------------------------------------------------------------


def scenario_2_broken_handoff() -> None:
    """Gatherer silently drops ``issues.open_p0``. Contract catches it."""
    divider("SCENARIO 2: Broken Handoff (Caught Before Release)")

    print(
        f"{DIM}Simulating an upstream refactor: the gatherer tool has been {RESET}\n"
        f"{DIM}updated so its 'issues' block no longer has 'open_p0' / 'open_p1'.{RESET}"
    )
    print(f"{DIM}Downstream agents keep running — the LLM papers over it —{RESET}")
    print(f"{DIM}but reagent-flow's handoff contract will fail the test.{RESET}\n")

    trace_dir = tempfile.mkdtemp()
    try:
        result = run_pipeline(
            "v2.3.1",
            trace_dir=trace_dir,
            drifted_gatherer=True,
        )

        try:
            result.assessor.assert_handoff_matches(schema=RELEASE_INFO_SCHEMA)
            # If we reach here, the "drift" was somehow harmless.
            print(
                f"{YELLOW}Unexpected: handoff still matches the contract. "
                f"The drifted gatherer payload was:{RESET}"
            )
            print(f"  {result.release_info}")
        except AssertionError as e:
            print(f"{RED}Handoff contract FAILED at gatherer -> assessor boundary:{RESET}\n")
            print(str(e))
            print()
            print(f"{YELLOW}reagent-flow caught the drift.{RESET}")
            print(
                f"{DIM}In CI this fails the test — the PR that introduced the "
                f"refactor can't merge until the contract is updated or the "
                f"gatherer is fixed.{RESET}"
            )
    finally:
        shutil.rmtree(trace_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Scenario 3 — Regression diff on the assessor session
# ---------------------------------------------------------------------------


def scenario_3_diff_path(trace_dir: str) -> None:
    """Run the pipeline with clean release data and diff against the golden."""
    divider("SCENARIO 3: Regression Detection (Baseline Diff)")

    print(f"{DIM}Using golden baselines from Scenario 1 (v2.3.1 — BLOCKED){RESET}")
    print(f"{DIM}Now running the same pipeline against v2.4.0 — agent should APPROVE{RESET}\n")

    result = run_pipeline("v2.4.0", trace_dir=trace_dir)

    print(f"{DIM}Diffing the assessor session against its golden baseline...{RESET}\n")
    try:
        result.assessor.assert_matches_baseline(
            ignore_fields={"arguments", "results", "response_text"}
        )
        print(f"{GREEN}Assessor trace matches baseline — tool flow unchanged.{RESET}")
    except AssertionError as e:
        print(f"{ORANGE}Baseline diff detected on the assessor session:{RESET}\n")
        print(str(e))
        print(f"\n{YELLOW}The assessor's tool-calling sequence changed between releases.{RESET}")
        print(
            f"{DIM}Here the change is expected (different input data). In real "
            f"CI this would flag an unintended behavioral regression from a "
            f"prompt tweak or model upgrade.{RESET}"
        )

    final = result.decision.get("decision", "?")
    print(f"\n{DIM}Final decision for v2.4.0: {final}{RESET}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all demo scenarios."""
    if not os.environ.get("GOOGLE_API_KEY"):
        print(f"{RED}GOOGLE_API_KEY not set.{RESET}")
        print("Get a free key at: https://aistudio.google.com/apikey")
        print("Then run: GOOGLE_API_KEY='your-key' uv run python demo.py")
        sys.exit(1)

    print(f"\n{BOLD}reagent-flow + LangGraph Demo: Release Risk Gatekeeper{RESET}")
    print(
        f"{DIM}Three sub-agents handing off structured data. "
        f"reagent-flow contracts catch drift at every boundary.{RESET}"
    )

    # Shared trace dir for scenarios 1 & 3 (golden baselines).
    trace_dir = tempfile.mkdtemp()

    try:
        scenario_1_green_path(trace_dir)
        scenario_2_broken_handoff()
        scenario_3_diff_path(trace_dir)

        divider("SUMMARY")
        print("reagent-flow treats every handoff as a contract.")
        print()
        print(f"  1. {GREEN}Green path{RESET}         — all contracts hold end-to-end")
        print(f"  2. {RED}Broken handoff{RESET}     — schema drift caught at gatherer -> assessor")
        print(f"  3. {ORANGE}Diff path{RESET}          — baseline regression detection")
        print()
        print(f"{DIM}Your agents are non-deterministic. Your contracts shouldn't be.{RESET}")
        print()
    finally:
        shutil.rmtree(trace_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
