"""reagent-flow: Behavioral testing library for AI agent tool-calling loops."""

from typing import Any

from reagent_flow.exceptions import (
    AmbiguousToolCallError,
    ReagentAdapterWarning,
    ReagentError,
    ReagentWarning,
    SessionClosedError,
    TraceNotFoundError,
)
from reagent_flow.session import Session

__version__ = "0.4.0"

__all__ = [
    "AmbiguousToolCallError",
    "Session",
    "SessionClosedError",
    "ReagentAdapterWarning",
    "ReagentError",
    "ReagentWarning",
    "TraceNotFoundError",
    "__version__",
    "session",
]


def session(
    name: str,
    *,
    golden: bool = False,
    metadata: dict[str, Any] | None = None,
    trace_dir: str = ".reagent",
    parent_trace_id: str | None = None,
    handoff_context: dict[str, Any] | None = None,
) -> Session:
    """Create a new recording session."""
    return Session(
        name,
        golden=golden,
        metadata=metadata,
        trace_dir=trace_dir,
        parent_trace_id=parent_trace_id,
        handoff_context=handoff_context,
    )
