"""CrewAI adapter for ttrace-ai."""

from __future__ import annotations

import warnings
from functools import wraps
from typing import Any

from ttrace_ai._context import get_active_session
from ttrace_ai.exceptions import TTraceAdapterWarning


def _warn(msg: str) -> None:
    warnings.warn(TTraceAdapterWarning(msg), stacklevel=3)


def instrument(crew: Any) -> Any:
    """Instrument a CrewAI Crew to capture tool calls into the active session.

    Wraps the crew's kickoff method and all agent tools at patch time.
    """
    agents = getattr(crew, "agents", [])
    for agent in agents:
        _wrap_agent_tools(agent)

    original_kickoff = crew.kickoff

    @wraps(original_kickoff)
    def wrapped_kickoff(*args: Any, **kwargs: Any) -> Any:
        return original_kickoff(*args, **kwargs)

    crew.kickoff = wrapped_kickoff
    return crew


def _wrap_agent_tools(agent: Any) -> None:
    """Wrap an agent's tools to capture calls."""
    tools = getattr(agent, "tools", None)
    if not tools:
        return

    for i, tool in enumerate(tools):
        if hasattr(tool, "_ttrace_wrapped"):
            continue

        original_run = getattr(tool, "_run", None)
        if original_run is None:
            continue

        tool_name = getattr(tool, "name", f"tool_{i}")

        @wraps(original_run)
        def make_wrapper(orig_run: Any, name: str) -> Any:
            def wrapped_run(*args: Any, **kwargs: Any) -> Any:
                session = get_active_session()
                if session is not None:
                    try:
                        session.log_llm_call(
                            tool_calls=[{"name": name, "arguments": dict(kwargs)}],
                        )
                    except Exception as e:
                        _warn(f"Failed to log CrewAI tool call: {e}")

                try:
                    result = orig_run(*args, **kwargs)
                except Exception as exc:
                    if session is not None:
                        try:
                            session.log_tool_result(name, error=str(exc))
                        except Exception as e:
                            _warn(f"Failed to log CrewAI tool error: {e}")
                    raise

                if session is not None:
                    try:
                        session.log_tool_result(name, result=str(result))
                    except Exception as e:
                        _warn(f"Failed to log CrewAI tool result: {e}")
                return result

            return wrapped_run

        tool._run = make_wrapper(original_run, tool_name)
        tool._ttrace_wrapped = True
