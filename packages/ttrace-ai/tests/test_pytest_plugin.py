"""Tests for pytest plugin."""


def test_plugin_registers_marker(pytestconfig: object) -> None:
    """Verify the plugin can be imported and has pytest_configure."""
    from ttrace_ai.pytest_plugin import pytest_configure

    assert callable(pytest_configure)


def test_plugin_adds_cli_options() -> None:
    """Verify the plugin has pytest_addoption."""
    from ttrace_ai.pytest_plugin import pytest_addoption

    assert callable(pytest_addoption)
