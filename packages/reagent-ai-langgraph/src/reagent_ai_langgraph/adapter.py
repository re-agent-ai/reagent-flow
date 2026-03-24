"""LangGraph adapter for reagent-ai.

Extends the LangChain callback handler with node transition tracking.
"""

from __future__ import annotations

import warnings
from typing import Any
from uuid import UUID

from reagent_ai.exceptions import ReagentAdapterWarning
from reagent_ai_langchain.handler import ReagentCallbackHandler


class ReagentGraphTracer(ReagentCallbackHandler):
    """LangGraph tracer — extends LangChain handler with node tracking.

    MVP limitation: node metadata (which graph node triggered a tool call) is
    tracked internally but not persisted to the trace. v0.2 will add
    Turn.metadata to support this. For MVP, this adapter provides the same
    capture as the LangChain adapter plus chain_start tracking for future use.
    """

    def __init__(self) -> None:
        super().__init__()
        self._current_node: str | None = None

    def on_chain_start(
        self,
        serialized: Any = None,
        inputs: Any = None,
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Track which graph node is currently executing."""
        try:
            if isinstance(serialized, dict) and "name" in serialized:
                self._current_node = serialized["name"]
        except Exception as e:
            warnings.warn(
                ReagentAdapterWarning(f"LangGraph node capture failed: {e}"),
                stacklevel=2,
            )
