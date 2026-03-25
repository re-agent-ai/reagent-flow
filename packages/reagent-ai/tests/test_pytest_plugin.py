"""Tests for pytest plugin fixtures and CLI flags."""

import pytest


def test_plugin_registers_marker(pytestconfig: object) -> None:
    """Verify the plugin can be imported and has pytest_configure."""
    from reagent_ai.pytest_plugin import pytest_configure

    assert callable(pytest_configure)


def test_plugin_adds_cli_options() -> None:
    """Verify the plugin has pytest_addoption."""
    from reagent_ai.pytest_plugin import pytest_addoption

    assert callable(pytest_addoption)


def test_reagent_dir_fixture(reagent_dir: str) -> None:
    """Verify reagent_dir fixture returns a string (defaults to .reagent)."""
    assert isinstance(reagent_dir, str)
    assert reagent_dir == ".reagent"


def test_reagent_record_fixture(reagent_record: bool) -> None:
    """Verify reagent_record fixture returns False by default."""
    assert reagent_record is False


def test_reagent_update_fixture(reagent_update: bool) -> None:
    """Verify reagent_update fixture returns False by default."""
    assert reagent_update is False


def test_reagent_session_fixture(reagent_session: object) -> None:
    """Verify reagent_session fixture provides a usable Session."""
    from reagent_ai.session import Session

    assert isinstance(reagent_session, Session)


@pytest.mark.reagent(golden=False)
def test_reagent_session_with_marker(reagent_session: object) -> None:
    """Verify reagent_session works with the @reagent marker."""
    from reagent_ai.session import Session

    assert isinstance(reagent_session, Session)
