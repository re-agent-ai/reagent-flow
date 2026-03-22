"""pytest plugin for ttrace-ai: CLI flags and markers."""

from __future__ import annotations

from typing import Any


def pytest_addoption(parser: Any) -> None:
    """Add ttrace-ai CLI options to pytest."""
    group = parser.getgroup("ttrace-ai", "TTrace-AI options")
    group.addoption(
        "--ttrace-record",
        action="store_true",
        default=False,
        help="Force live recording (ignore cached traces)",
    )
    group.addoption(
        "--ttrace-update",
        action="store_true",
        default=False,
        help="Re-record golden baselines",
    )
    group.addoption(
        "--ttrace-dir",
        default=".ttrace",
        help="Override .ttrace/ directory location",
    )


def pytest_configure(config: Any) -> None:
    """Register ttrace-ai markers."""
    config.addinivalue_line(
        "markers",
        "ttrace(golden=False): mark test as a ttrace-ai test",
    )
