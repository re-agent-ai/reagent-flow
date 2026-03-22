"""Tests for CrewAI adapter."""

from unittest.mock import MagicMock

import ttrace_ai
from ttrace_ai_crewai import instrument


def _mock_crew() -> MagicMock:
    """Create a mock CrewAI crew with one agent and one tool."""
    tool = MagicMock()
    tool.name = "search"
    tool._run = MagicMock(return_value="found it")
    tool._ttrace_wrapped = False
    del tool._ttrace_wrapped  # Remove so hasattr returns False

    agent = MagicMock()
    agent.tools = [tool]

    crew = MagicMock()
    crew.agents = [agent]
    crew.kickoff = MagicMock(return_value="crew result")

    return crew


def test_instrument_returns_crew() -> None:
    crew = _mock_crew()
    result = instrument(crew)
    assert result is crew


def test_instrument_wraps_kickoff() -> None:
    crew = _mock_crew()
    original = crew.kickoff
    instrument(crew)
    assert crew.kickoff is not original


def test_instrument_captures_tool_calls() -> None:
    crew = _mock_crew()
    instrument(crew)

    with ttrace_ai.session("test") as s:
        crew.kickoff()
        # Simulate the tool being called during kickoff
        tool = crew.agents[0].tools[0]
        tool._run(query="test query")

    assert len(s.trace.turns) >= 1


def test_instrument_noop_without_session() -> None:
    crew = _mock_crew()
    instrument(crew)
    crew.kickoff()
    # Should not raise
