"""Tests for LangChain adapter."""

from unittest.mock import MagicMock
from uuid import uuid4

import ttrace_ai
from ttrace_ai_langchain import TTraceCallbackHandler


def test_handler_captures_llm_with_tool_calls() -> None:
    handler = TTraceCallbackHandler()

    response = MagicMock()
    gen = MagicMock()
    gen.text = None
    tc = {"name": "lookup", "args": {"id": "1"}, "id": "call_1"}
    gen.message.tool_calls = [tc]
    response.generations = [[gen]]

    with ttrace_ai.session("test") as s:
        handler.on_llm_end(response, run_id=uuid4())

    assert len(s.trace.turns) == 1
    assert s.trace.turns[0].llm_call.tool_calls[0].name == "lookup"


def test_handler_captures_tool_result() -> None:
    handler = TTraceCallbackHandler()

    response = MagicMock()
    gen = MagicMock()
    gen.text = None
    gen.message.tool_calls = [{"name": "lookup", "args": {}, "id": "c1"}]
    response.generations = [[gen]]

    with ttrace_ai.session("test") as s:
        handler.on_llm_end(response, run_id=uuid4())
        handler.on_tool_end("result data", run_id=uuid4(), name="lookup")

    assert len(s.trace.turns[0].tool_results) == 1
    assert s.trace.turns[0].tool_results[0].result == "result data"


def test_handler_captures_tool_error() -> None:
    handler = TTraceCallbackHandler()

    response = MagicMock()
    gen = MagicMock()
    gen.text = None
    gen.message.tool_calls = [{"name": "lookup", "args": {}, "id": "c1"}]
    response.generations = [[gen]]

    with ttrace_ai.session("test") as s:
        handler.on_llm_end(response, run_id=uuid4())
        handler.on_tool_error(ValueError("not found"), run_id=uuid4(), name="lookup")

    assert s.trace.turns[0].tool_results[0].error == "not found"


def test_handler_noop_without_session() -> None:
    handler = TTraceCallbackHandler()
    response = MagicMock()
    gen = MagicMock()
    gen.text = "hello"
    gen.message.tool_calls = []
    response.generations = [[gen]]
    handler.on_llm_end(response, run_id=uuid4())
    # Should not raise


def test_handler_parallel_same_name_tool_calls() -> None:
    """Handler should correctly correlate results to same-name tool calls via call_id."""
    handler = TTraceCallbackHandler()

    response = MagicMock()
    gen = MagicMock()
    gen.text = None
    gen.message.tool_calls = [
        {"name": "lookup", "args": {"id": "1"}, "id": "call_a"},
        {"name": "lookup", "args": {"id": "2"}, "id": "call_b"},
    ]
    response.generations = [[gen]]

    with ttrace_ai.session("test") as s:
        handler.on_llm_end(response, run_id=uuid4())
        handler.on_tool_end("result_1", run_id=uuid4(), name="lookup")
        handler.on_tool_end("result_2", run_id=uuid4(), name="lookup")

    results = s.trace.turns[0].tool_results
    assert len(results) == 2
    assert results[0].result == "result_1"
    assert results[1].result == "result_2"
