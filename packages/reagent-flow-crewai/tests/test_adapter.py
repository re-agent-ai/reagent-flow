"""Tests for CrewAI adapter."""

from unittest.mock import MagicMock

import reagent_flow
from reagent_flow_crewai import instrument


def _mock_crew() -> MagicMock:
    """Create a mock CrewAI crew with one agent and one tool."""
    tool = MagicMock()
    tool.name = "search"
    tool._run = MagicMock(return_value="found it")
    tool._reagent_wrapped = False
    del tool._reagent_wrapped  # Remove so hasattr returns False

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

    with reagent_flow.session("test") as s:
        crew.kickoff()
        # Simulate the tool being called during kickoff
        tool = crew.agents[0].tools[0]
        tool._run(query="test query")

    assert len(s.trace.turns) >= 1


def test_instrument_preserves_structured_result() -> None:
    """P2 fix: Tool results should be stored as-is, not coerced to str."""
    structured = {"status": "found", "count": 42}
    crew = _mock_crew()
    # Replace _run with one that returns structured data
    crew.agents[0].tools[0]._run.return_value = structured
    instrument(crew)

    tool = crew.agents[0].tools[0]
    with reagent_flow.session("test") as s:
        tool._run(query="test")

    result = s.trace.turns[0].tool_results[0].result
    assert result == structured
    assert isinstance(result, dict)


def test_instrument_captures_positional_args() -> None:
    """P2 fix: Positional args should be recorded, not silently dropped."""
    crew = _mock_crew()
    instrument(crew)

    tool = crew.agents[0].tools[0]
    with reagent_flow.session("test") as s:
        tool._run("pos1", "pos2", key="val")

    args = s.trace.turns[0].llm_call.tool_calls[0].arguments
    assert args["_positional"] == ["pos1", "pos2"]
    assert args["key"] == "val"


def test_instrument_noop_without_session() -> None:
    crew = _mock_crew()
    instrument(crew)
    crew.kickoff()
    # Should not raise
