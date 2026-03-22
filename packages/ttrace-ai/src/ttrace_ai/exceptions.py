"""TTrace-AI exceptions and warnings."""


class TTraceError(Exception):
    """Base exception for ttrace-ai."""


class SessionClosedError(TTraceError):
    """Raised when log_* is called on a finalized session."""


class AmbiguousToolCallError(TTraceError):
    """Raised when log_tool_result can't disambiguate parallel tool calls."""


class TraceNotFoundError(TTraceError, FileNotFoundError):
    """Raised when a referenced trace or golden baseline doesn't exist."""


class TTraceAdapterWarning(UserWarning):
    """Emitted when an adapter fails to capture data."""
