"""Deterministic product showcase for reagent-flow.

Scenario: an AI vendor-onboarding workflow is about to approve a SaaS vendor
that handles customer data. The intake agent drifts its payload, the security
agent keeps going, and reagent-flow catches the broken handoff before approval.

Run with:
    cd examples/vendor_onboarding_showcase
    uv run python showcase.py
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from typing import Any

import reagent_flow

GREEN = "\033[32m"
RED = "\033[31m"
ORANGE = "\033[38;5;208m"
YELLOW = "\033[38;5;220m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

VENDOR_INTAKE_SCHEMA = {
    "vendor_name": str,
    "requesting_team": str,
    "business_purpose": str,
    "annual_cost_usd": int,
    "data_access": {
        "contains_customer_pii": bool,
        "data_categories": [str],
        "storage_region": str,
        "retention_days": int,
    },
    "compliance": {
        "soc2_available": bool,
        "dpa_required": bool,
        "subprocessors": [str],
    },
    "contract": {
        "term_months": int,
        "auto_renewal": bool,
        "termination_notice_days": int,
    },
}

SECURITY_REVIEW_SCHEMA = {
    "vendor_name": str,
    "security_risk": str,
    "requires_dpa": bool,
    "control_gaps": [str],
}

FINANCE_REVIEW_SCHEMA = {
    "vendor_name": str,
    "budget_status": str,
    "annual_cost_usd": int,
    "renewal_risk": str,
}

APPROVAL_PACKET_SCHEMA = {
    "vendor_name": str,
    "security_risk": str,
    "budget_status": str,
    "decision": str,
    "reason": str,
}

VENDOR_REQUEST: dict[str, Any] = {
    "vendor_name": "ClearVoice AI",
    "requesting_team": "Customer Success",
    "business_purpose": "Transcribe customer calls and summarize follow-up actions.",
    "annual_cost_usd": 48000,
    "data_access": {
        "contains_customer_pii": True,
        "data_categories": ["call_recordings", "email_addresses", "account_ids"],
        "storage_region": "US",
        "retention_days": 365,
    },
    "compliance": {
        "soc2_available": True,
        "dpa_required": True,
        "subprocessors": ["AcmeCloud", "VectorStore Inc"],
    },
    "contract": {
        "term_months": 12,
        "auto_renewal": True,
        "termination_notice_days": 60,
    },
}


@dataclass
class PipelineResult:
    """Sessions and payloads produced by the deterministic showcase."""

    intake: reagent_flow.Session
    security: reagent_flow.Session
    finance: reagent_flow.Session
    approver: reagent_flow.Session
    vendor_packet: dict[str, Any]
    security_review: dict[str, Any]
    finance_review: dict[str, Any]
    approval: dict[str, Any]


def drift_vendor_packet(packet: dict[str, Any]) -> dict[str, Any]:
    """Simulate a realistic upstream extraction drift in the intake agent."""
    drifted = {
        key: value.copy() if isinstance(value, dict) else value for key, value in packet.items()
    }
    drifted["data_access"] = {
        "handles_personal_data": packet["data_access"]["contains_customer_pii"],
        "data_categories": packet["data_access"]["data_categories"],
        "storage_region": packet["data_access"]["storage_region"],
        "retention_days": packet["data_access"]["retention_days"],
    }
    drifted["compliance"] = {
        "soc2_available": packet["compliance"]["soc2_available"],
        "dpa_required": packet["compliance"]["dpa_required"],
        "subprocessors": ", ".join(packet["compliance"]["subprocessors"]),
    }
    return drifted


def _security_review_from(packet: dict[str, Any]) -> dict[str, Any]:
    data_access = packet.get("data_access", {})
    compliance = packet.get("compliance", {})
    contains_pii = bool(data_access.get("contains_customer_pii", False))
    subprocessors = compliance.get("subprocessors", [])
    control_gaps = []
    if contains_pii and not compliance.get("dpa_required", False):
        control_gaps.append("DPA required before approval")
    if contains_pii and not isinstance(subprocessors, list):
        control_gaps.append("Subprocessors must be reviewed as a list")

    return {
        "vendor_name": str(packet.get("vendor_name", "unknown")),
        "security_risk": "HIGH" if contains_pii else "LOW",
        "requires_dpa": contains_pii,
        "control_gaps": control_gaps,
    }


def _finance_review_from(packet: dict[str, Any]) -> dict[str, Any]:
    annual_cost = int(packet.get("annual_cost_usd", 0))
    contract = packet.get("contract", {})
    return {
        "vendor_name": str(packet.get("vendor_name", "unknown")),
        "budget_status": "IN_BUDGET" if annual_cost <= 50000 else "NEEDS_APPROVAL",
        "annual_cost_usd": annual_cost,
        "renewal_risk": "HIGH" if contract.get("auto_renewal") else "LOW",
    }


def _approval_from(
    security_review: dict[str, Any],
    finance_review: dict[str, Any],
) -> dict[str, Any]:
    approved = (
        security_review["security_risk"] == "LOW" and finance_review["budget_status"] == "IN_BUDGET"
    )
    return {
        "vendor_name": security_review["vendor_name"],
        "security_risk": security_review["security_risk"],
        "budget_status": finance_review["budget_status"],
        "decision": "APPROVE" if approved else "ESCALATE",
        "reason": (
            "Security and budget checks passed."
            if approved
            else "Security or budget review requires human approval."
        ),
    }


def run_pipeline(trace_dir: str, *, drifted_intake: bool) -> PipelineResult:
    """Run the deterministic vendor-onboarding agent workflow."""
    vendor_packet = drift_vendor_packet(VENDOR_REQUEST) if drifted_intake else VENDOR_REQUEST

    with reagent_flow.session("vendor-intake", trace_dir=trace_dir) as intake:
        intake.log_llm_call(
            tool_calls=[
                {
                    "name": "extract_vendor_packet",
                    "arguments": {"request_id": "VR-2026-0417"},
                }
            ],
        )
        intake.log_tool_result("extract_vendor_packet", result=vendor_packet)
        intake.log_llm_call(response_text="Vendor packet extracted.", tool_calls=[])

    security_review = _security_review_from(vendor_packet)
    with reagent_flow.session(
        "vendor-security-review",
        trace_dir=trace_dir,
        parent_trace_id=intake.trace.trace_id,
        handoff_context=vendor_packet,
    ) as security:
        security.log_llm_call(
            tool_calls=[
                {
                    "name": "review_security",
                    "arguments": {
                        "vendor_name": vendor_packet.get("vendor_name", "unknown"),
                    },
                }
            ],
        )
        security.log_tool_result("review_security", result=security_review)
        security.log_llm_call(response_text="Security review completed.", tool_calls=[])

    finance_review = _finance_review_from(vendor_packet)
    with reagent_flow.session(
        "vendor-finance-review",
        trace_dir=trace_dir,
        parent_trace_id=intake.trace.trace_id,
        handoff_context=vendor_packet,
    ) as finance:
        finance.log_llm_call(
            tool_calls=[
                {
                    "name": "review_budget",
                    "arguments": {
                        "vendor_name": vendor_packet.get("vendor_name", "unknown"),
                    },
                }
            ],
        )
        finance.log_tool_result("review_budget", result=finance_review)
        finance.log_llm_call(response_text="Finance review completed.", tool_calls=[])

    approval = _approval_from(security_review, finance_review)
    with reagent_flow.session(
        "vendor-approval",
        trace_dir=trace_dir,
        parent_trace_id=security.trace.trace_id,
        handoff_context=approval,
    ) as approver:
        approver.log_llm_call(
            tool_calls=[
                {
                    "name": "record_vendor_decision",
                    "arguments": {
                        "vendor_name": approval["vendor_name"],
                        "decision": approval["decision"],
                    },
                }
            ],
        )
        approver.log_tool_result("record_vendor_decision", result=approval)
        approver.log_llm_call(response_text="Vendor decision recorded.", tool_calls=[])

    return PipelineResult(
        intake=intake,
        security=security,
        finance=finance,
        approver=approver,
        vendor_packet=vendor_packet,
        security_review=security_review,
        finance_review=finance_review,
        approval=approval,
    )


def assert_green_path(result: PipelineResult) -> None:
    """Assert the complete workflow contract for a valid vendor packet."""
    result.intake.assert_tool_output_matches(
        "extract_vendor_packet",
        schema=VENDOR_INTAKE_SCHEMA,
    )
    result.security.assert_handoff_received(result.intake)
    result.security.assert_handoff_matches(schema=VENDOR_INTAKE_SCHEMA)
    result.security.assert_tool_output_matches("review_security", schema=SECURITY_REVIEW_SCHEMA)
    result.finance.assert_handoff_received(result.intake)
    result.finance.assert_handoff_matches(schema=VENDOR_INTAKE_SCHEMA)
    result.finance.assert_tool_output_matches("review_budget", schema=FINANCE_REVIEW_SCHEMA)
    result.approver.assert_handoff_received(result.security)
    result.approver.assert_handoff_matches(schema=APPROVAL_PACKET_SCHEMA)
    result.approver.assert_tool_output_matches(
        "record_vendor_decision",
        schema=APPROVAL_PACKET_SCHEMA,
    )


def _divider(title: str) -> None:
    width = 76
    print(f"\n{ORANGE}{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}{RESET}\n")


def main() -> None:
    """Run the deterministic vendor-onboarding showcase."""
    trace_dir = tempfile.mkdtemp()
    try:
        _divider("reagent-flow Showcase: AI Vendor Approval Blocked Before Risk Review")
        print(
            f"{DIM}Workflow:{RESET} Intake -> Security Review + Finance Review -> Approver\n"
            f"{DIM}Scenario:{RESET} ClearVoice AI will transcribe customer calls\n"
        )

        print(f"{BOLD}1. The contract{RESET}")
        print("   Security expects a complete vendor packet:")
        print("   - data_access.contains_customer_pii: bool")
        print("   - compliance.subprocessors: list[str]\n")

        print(f"{BOLD}2. The upstream drift{RESET}")
        print("   Intake still emits a vendor packet, but changed:")
        print("   - contains_customer_pii -> handles_personal_data")
        print("   - subprocessors list -> comma-separated string\n")

        result = run_pipeline(trace_dir, drifted_intake=True)
        print(f"{BOLD}3. What would happen without the contract{RESET}")
        print(f"   Security risk: {YELLOW}{result.security_review['security_risk']}{RESET}")
        print(f"   Approval decision: {YELLOW}{result.approval['decision']}{RESET}\n")

        print(f"{BOLD}4. reagent-flow handoff assertion{RESET}")
        try:
            result.security.assert_handoff_matches(schema=VENDOR_INTAKE_SCHEMA)
        except AssertionError as exc:
            print(f"{RED}FAILED test_vendor_onboarding.py::test_intake_to_security_contract{RESET}")
            print(str(exc))
            print(
                f"\n{GREEN}Result:{RESET} vendor approval blocked before a risky "
                "security review reaches the approver."
            )
            return

        print("Unexpected: drifted vendor handoff still matched the contract.")
    finally:
        shutil.rmtree(trace_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
