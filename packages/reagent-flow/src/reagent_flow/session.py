"""Session context manager for recording and asserting on agent traces."""

from __future__ import annotations

import contextvars
import time
import types
import uuid
from types import EllipsisType
from typing import Any

from reagent_flow._context import _active_session
from reagent_flow.exceptions import SessionClosedError
from reagent_flow.models import Trace
from reagent_flow.recorder import Recorder


class Session:
    """Context manager for recording and asserting on agent traces."""

    def __init__(
        self,
        name: str,
        *,
        golden: bool = False,
        metadata: dict[str, Any] | None = None,
        trace_dir: str = ".reagent",
        parent_trace_id: str | None = None,
        handoff_context: dict[str, Any] | None = None,
    ) -> None:
        self.trace = Trace(
            trace_id=str(uuid.uuid4()),
            name=name,
            metadata=metadata or {},
            started_at=time.time(),
            parent_trace_id=parent_trace_id,
            handoff_context=handoff_context,
        )
        self._golden = golden
        self._trace_dir = trace_dir
        self._recorder = Recorder()
        self._closed = False
        self._token: contextvars.Token[Session | None] | None = None

    def __enter__(self) -> Session:
        self._token = _active_session.set(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        self._finalize()

    def _finalize(self) -> None:
        """Shared teardown for sync and async context managers."""
        self._sync_trace()
        self.trace.ended_at = time.time()
        self._closed = True
        if self._token is not None:
            _active_session.reset(self._token)
            self._token = None
        self._save()

    def _sync_trace(self) -> None:
        """Synchronize trace.turns from the recorder's live state."""
        self.trace.turns = self._recorder.turns

    def _save(self) -> None:
        """Save trace to storage."""
        from reagent_flow.storage.json import save_trace

        save_trace(self.trace, self._trace_dir, golden=self._golden)

    def log_llm_call(self, **kwargs: Any) -> list[str]:
        """Log an LLM call. Returns list of call_ids."""
        if self._closed:
            raise SessionClosedError("Cannot log to a finalized session.")
        return self._recorder.log_llm_call(**kwargs)

    def log_tool_result(self, name: str, **kwargs: Any) -> None:
        """Log a tool result."""
        if self._closed:
            raise SessionClosedError("Cannot log to a finalized session.")
        self._recorder.log_tool_result(name, **kwargs)

    def assert_called(self, tool_name: str) -> None:
        """Assert that a tool was called at least once."""
        from reagent_flow.assertions import assert_called

        self._sync_trace()
        assert_called(self.trace, tool_name)

    def assert_never_called(self, tool_name: str) -> None:
        """Assert that a tool was never called."""
        from reagent_flow.assertions import assert_never_called

        self._sync_trace()
        assert_never_called(self.trace, tool_name)

    def assert_called_before(self, first: str, second: str) -> None:
        """Assert that first tool was called before second tool."""
        from reagent_flow.assertions import assert_called_before

        self._sync_trace()
        assert_called_before(self.trace, first, second)

    def assert_tool_succeeded(self, tool_name: str) -> None:
        """Assert that a tool was called and succeeded."""
        from reagent_flow.assertions import assert_tool_succeeded

        self._sync_trace()
        assert_tool_succeeded(self.trace, tool_name)

    def assert_max_turns(self, n: int) -> None:
        """Assert that the trace has at most n turns."""
        from reagent_flow.assertions import assert_max_turns

        self._sync_trace()
        assert_max_turns(self.trace, n)

    def assert_total_duration_under(self, *, ms: float) -> None:
        """Assert total trace duration is under ms milliseconds."""
        from reagent_flow.assertions import assert_total_duration_under

        self._sync_trace()
        assert_total_duration_under(self.trace, ms=ms)

    def assert_matches_baseline(
        self,
        *,
        base_dir: str | None = None,
        ignore_fields: set[str] | None = None,
    ) -> None:
        """Assert that the current trace matches its golden baseline.

        Args:
            base_dir: Override trace directory for loading the golden baseline.
            ignore_fields: Set of field paths to ignore during comparison.
                Supported: ``"arguments"``, ``"results"``, ``"response_text"``,
                or specific keys like ``"tool_name.arg_key"``.

        """
        from reagent_flow.diff import diff_traces
        from reagent_flow.storage.json import load_golden

        self._sync_trace()
        dir_path = base_dir or self._trace_dir
        golden = load_golden(dir_path, self.trace.name)
        result = diff_traces(golden, self.trace, ignore_fields=ignore_fields)
        if not result.is_match:
            raise AssertionError(f"Trace does not match golden baseline:\n{result.summary()}")

    def assert_flow(self, pattern: list[str | EllipsisType]) -> None:
        """Assert that tool calls match a flow pattern."""
        from reagent_flow.assertions import assert_flow

        self._sync_trace()
        assert_flow(self.trace, pattern)

    def assert_called_times(self, tool_name: str, *, min: int = 0, max: int | None = None) -> None:
        """Assert that a tool was called between min and max times."""
        from reagent_flow.assertions import assert_called_times

        self._sync_trace()
        assert_called_times(self.trace, tool_name, min=min, max=max)

    def assert_called_with(self, tool_name: str, **expected_args: Any) -> None:
        """Assert that a tool was called with specific argument values."""
        from reagent_flow.assertions import assert_called_with

        self._sync_trace()
        assert_called_with(self.trace, tool_name, **expected_args)

    def assert_handoff_received(self, parent: Session) -> None:
        """Assert that this session's trace is linked to a parent session."""
        from reagent_flow.assertions import assert_handoff_received

        self._sync_trace()
        parent._sync_trace()
        assert_handoff_received(self.trace, parent.trace)

    def assert_handoff_has_fields(self, fields: list[str]) -> None:
        """Assert that required fields exist in handoff context."""
        from reagent_flow.assertions import assert_handoff_has_fields

        self._sync_trace()
        assert_handoff_has_fields(self.trace, fields=fields)

    def assert_total_tokens_under(self, n: int, *, allow_missing: bool = False) -> None:
        """Assert that total token usage is under n."""
        from reagent_flow.assertions import assert_total_tokens_under

        self._sync_trace()
        assert_total_tokens_under(self.trace, n, allow_missing=allow_missing)

    def assert_cost_under(
        self,
        *,
        usd: float,
        model_costs: dict[str, dict[str, float]],
        allow_unpriced: bool = False,
    ) -> None:
        """Assert that estimated cost is under a USD limit."""
        from reagent_flow.assertions import assert_cost_under

        self._sync_trace()
        assert_cost_under(
            self.trace, usd=usd, model_costs=model_costs, allow_unpriced=allow_unpriced
        )

    def assert_handoff_matches(
        self, *, schema: dict[str, type | dict[str, Any] | list[Any]]
    ) -> None:
        """Assert that handoff_context matches a schema."""
        from reagent_flow.assertions import assert_handoff_matches

        self._sync_trace()
        assert_handoff_matches(self.trace, schema=schema)

    def assert_no_extra_fields(self, *, allowed: list[str]) -> None:
        """Assert that handoff_context has no unexpected fields."""
        from reagent_flow.assertions import assert_no_extra_fields

        self._sync_trace()
        assert_no_extra_fields(self.trace, allowed=allowed)

    def assert_tool_output_matches(
        self, tool_name: str, *, schema: dict[str, type | dict[str, Any] | list[Any]]
    ) -> None:
        """Assert that tool results match a schema."""
        from reagent_flow.assertions import assert_tool_output_matches

        self._sync_trace()
        assert_tool_output_matches(self.trace, tool_name, schema=schema)

    def assert_context_preserved(self, source: dict[str, Any], *, fields: list[str]) -> None:
        """Assert that specific values survived a handoff."""
        from reagent_flow.assertions import assert_context_preserved

        self._sync_trace()
        assert_context_preserved(source, self.trace, fields=fields)

    # -- Async context manager support --

    async def __aenter__(self) -> Session:
        """Enter an async session context."""
        self._token = _active_session.set(self)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Exit an async session context."""
        self._finalize()
