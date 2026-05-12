"""Intentional failing pytest demo for screenshot/GIF proof.

This file is not named ``test_*.py`` so normal example test runs do not collect
it. Run it explicitly when you want to show the CI failure reagent-flow would
produce for a drifted multi-agent handoff:

    uv run pytest demo_failure_contract.py -v
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


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


def test_drifted_intake_handoff_fails_ci(tmp_path: Path) -> None:
    """Intentional failure: the intake agent drifted the security handoff."""
    result = showcase.run_pipeline(str(tmp_path), drifted_intake=True)

    result.security.assert_handoff_matches(schema=showcase.VENDOR_INTAKE_SCHEMA)
