"""Tests for the deterministic vendor-onboarding showcase."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_showcase() -> ModuleType:
    module_path = Path(__file__).with_name("showcase.py")
    spec = importlib.util.spec_from_file_location("vendor_onboarding_showcase", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load showcase module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


showcase = _load_showcase()


def test_green_path_vendor_onboarding_contracts_hold(tmp_path: Path) -> None:
    """A complete vendor packet satisfies every agent-boundary contract."""
    result = showcase.run_pipeline(str(tmp_path), drifted_intake=False)

    showcase.assert_green_path(result)


def test_intake_to_security_drift_is_caught(tmp_path: Path) -> None:
    """The security agent cannot receive a drifted vendor packet silently."""
    result = showcase.run_pipeline(str(tmp_path), drifted_intake=True)

    with pytest.raises(AssertionError) as error:
        result.security.assert_handoff_matches(schema=showcase.VENDOR_INTAKE_SCHEMA)

    message = str(error.value)
    assert "data_access.contains_customer_pii" in message
    assert "missing" in message


def test_drifted_packet_would_approve_without_contract(tmp_path: Path) -> None:
    """The business-risk punchline: the broken handoff creates a false approval."""
    result = showcase.run_pipeline(str(tmp_path), drifted_intake=True)

    assert result.security_review["security_risk"] == "LOW"
    assert result.approval["decision"] == "APPROVE"


def test_clean_packet_escalates_customer_pii_vendor(tmp_path: Path) -> None:
    """The clean packet preserves the PII flag, so security risk is not hidden."""
    result = showcase.run_pipeline(str(tmp_path), drifted_intake=False)

    assert result.security_review["security_risk"] == "HIGH"
    assert result.approval["decision"] == "ESCALATE"
