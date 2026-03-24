"""pytest plugin for reagent-ai: CLI flags and markers."""

from __future__ import annotations

from typing import Any


def pytest_addoption(parser: Any) -> None:
    """Add reagent-ai CLI options to pytest."""
    group = parser.getgroup("reagent-ai", "Reagent-AI options")
    group.addoption(
        "--reagent-record",
        action="store_true",
        default=False,
        help="Force live recording (ignore cached traces)",
    )
    group.addoption(
        "--reagent-update",
        action="store_true",
        default=False,
        help="Re-record golden baselines",
    )
    group.addoption(
        "--reagent-dir",
        default=".reagent",
        help="Override .reagent/ directory location",
    )


def pytest_configure(config: Any) -> None:
    """Register reagent-ai markers."""
    config.addinivalue_line(
        "markers",
        "reagent(golden=False): mark test as a reagent-ai test",
    )
