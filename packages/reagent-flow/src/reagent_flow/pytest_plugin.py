"""pytest plugin for reagent-flow: CLI flags, markers, and fixtures."""

from __future__ import annotations

from typing import Any, Generator

import pytest

from reagent_flow.session import Session


def pytest_addoption(parser: Any) -> None:
    """Add reagent-flow CLI options to pytest."""
    group = parser.getgroup("reagent-flow", "Reagent-Flow options")
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
    """Register reagent-flow markers."""
    config.addinivalue_line(
        "markers",
        "reagent(golden=False): mark test as a reagent-flow test",
    )


@pytest.fixture
def reagent_dir(request: pytest.FixtureRequest) -> str:
    """Return the configured trace directory."""
    return str(request.config.getoption("--reagent-dir"))


@pytest.fixture
def reagent_record(request: pytest.FixtureRequest) -> bool:
    """Return whether --reagent-record was passed."""
    return bool(request.config.getoption("--reagent-record"))


@pytest.fixture
def reagent_update(request: pytest.FixtureRequest) -> bool:
    """Return whether --reagent-update was passed."""
    return bool(request.config.getoption("--reagent-update"))


@pytest.fixture
def reagent_session(
    request: pytest.FixtureRequest,
) -> Generator[Session, None, None]:
    """Provide a managed Session for the current test.

    Reads all ``--reagent-*`` flags from the CLI:

    - ``--reagent-dir``: trace storage directory (default ``.reagent``)
    - ``--reagent-update``: re-record golden baselines
    - ``--reagent-record``: force live recording; sets ``reagent_record=True``
      in session metadata so adapters and test helpers can branch on it

    If the test is decorated with ``@pytest.mark.reagent(golden=True)``
    or ``--reagent-update`` is passed, the session records a golden baseline.
    """
    trace_dir: str = str(request.config.getoption("--reagent-dir"))
    update: bool = bool(request.config.getoption("--reagent-update"))
    record: bool = bool(request.config.getoption("--reagent-record"))

    # Determine golden flag from marker or CLI
    marker = request.node.get_closest_marker("reagent")
    golden = update
    if marker is not None:
        golden = golden or bool(marker.kwargs.get("golden", False))

    metadata: dict[str, object] = {}
    if record:
        metadata["reagent_record"] = True

    name = request.node.name
    with Session(name, golden=golden, trace_dir=trace_dir, metadata=metadata) as session:
        yield session
