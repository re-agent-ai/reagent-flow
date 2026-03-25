"""Reagent-AI exceptions and warnings."""


class ReagentError(Exception):
    """Base exception for reagent-ai."""


class SessionClosedError(ReagentError):
    """Raised when log_* is called on a finalized session."""


class AmbiguousToolCallError(ReagentError):
    """Raised when log_tool_result can't disambiguate parallel tool calls."""


class TraceNotFoundError(ReagentError, FileNotFoundError):
    """Raised when a referenced trace or golden baseline doesn't exist."""


class ReagentWarning(UserWarning):
    """General warning emitted by reagent-ai core."""


class ReagentAdapterWarning(UserWarning):
    """Emitted when an adapter fails to capture data."""
