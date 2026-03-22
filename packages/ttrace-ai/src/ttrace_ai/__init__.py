"""ttrace-ai: Behavioral testing library for AI agent tool-calling loops."""

from typing import Any

from ttrace_ai.exceptions import (
    AmbiguousToolCallError,
    SessionClosedError,
    TraceNotFoundError,
    TTraceAdapterWarning,
    TTraceError,
)
from ttrace_ai.session import Session

__version__ = "0.1.0"

__all__ = [
    "AmbiguousToolCallError",
    "Session",
    "SessionClosedError",
    "TTraceAdapterWarning",
    "TTraceError",
    "TraceNotFoundError",
    "__version__",
    "session",
]


def session(
    name: str,
    *,
    golden: bool = False,
    metadata: dict[str, Any] | None = None,
    trace_dir: str = ".ttrace",
) -> Session:
    """Create a new recording session."""
    return Session(name, golden=golden, metadata=metadata, trace_dir=trace_dir)
