"""Session context manager for recording and asserting on agent traces."""

from __future__ import annotations

import time
import uuid
from typing import Any

from reagent_ai._context import _active_session
from reagent_ai.exceptions import SessionClosedError
from reagent_ai.models import Trace
from reagent_ai.recorder import Recorder


class Session:
    """Context manager for recording and asserting on agent traces."""

    def __init__(
        self,
        name: str,
        *,
        golden: bool = False,
        metadata: dict[str, Any] | None = None,
        trace_dir: str = ".reagent",
    ) -> None:
        self.trace = Trace(
            trace_id=str(uuid.uuid4()),
            name=name,
            metadata=metadata or {},
            started_at=time.time(),
        )
        self._golden = golden
        self._trace_dir = trace_dir
        self._recorder = Recorder()
        self._closed = False
        self._token: Any = None

    def __enter__(self) -> Session:
        self._token = _active_session.set(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
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
        from reagent_ai.storage.json import save_trace

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
        from reagent_ai.assertions import assert_called

        self._sync_trace()
        assert_called(self.trace, tool_name)

    def assert_never_called(self, tool_name: str) -> None:
        """Assert that a tool was never called."""
        from reagent_ai.assertions import assert_never_called

        self._sync_trace()
        assert_never_called(self.trace, tool_name)

    def assert_called_before(self, first: str, second: str) -> None:
        """Assert that first tool was called before second tool."""
        from reagent_ai.assertions import assert_called_before

        self._sync_trace()
        assert_called_before(self.trace, first, second)

    def assert_tool_succeeded(self, tool_name: str) -> None:
        """Assert that a tool was called and succeeded."""
        from reagent_ai.assertions import assert_tool_succeeded

        self._sync_trace()
        assert_tool_succeeded(self.trace, tool_name)

    def assert_max_turns(self, n: int) -> None:
        """Assert that the trace has at most n turns."""
        from reagent_ai.assertions import assert_max_turns

        self._sync_trace()
        assert_max_turns(self.trace, n)

    def assert_total_duration_under(self, *, ms: float) -> None:
        """Assert total trace duration is under ms milliseconds."""
        from reagent_ai.assertions import assert_total_duration_under

        self._sync_trace()
        assert_total_duration_under(self.trace, ms=ms)

    def assert_matches_baseline(self, *, base_dir: str | None = None) -> None:
        """Assert that the current trace matches its golden baseline."""
        from reagent_ai.diff import diff_traces
        from reagent_ai.storage.json import load_golden

        self._sync_trace()
        dir_path = base_dir or self._trace_dir
        golden = load_golden(dir_path, self.trace.name)
        result = diff_traces(golden, self.trace)
        if not result.is_match:
            raise AssertionError(f"Trace does not match golden baseline:\n{result.summary()}")
