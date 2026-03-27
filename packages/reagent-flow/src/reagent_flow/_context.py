"""Context variable for active session binding."""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reagent_flow.session import Session

_active_session: ContextVar[Session | None] = ContextVar("reagent_flow_session", default=None)


def get_active_session() -> Session | None:
    """Return the active session, or None if no session is active."""
    return _active_session.get()
