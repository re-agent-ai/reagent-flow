"""Tests for LangChain adapter."""

from unittest.mock import MagicMock
from uuid import uuid4

import reagent_flow
from reagent_flow_langchain import ReagentCallbackHandler


def test_handler_captures_llm_with_tool_calls() -> None:
    handler = ReagentCallbackHandler()

    response = MagicMock()
    gen = MagicMock()
    gen.text = None
    tc = {"name": "lookup", "args": {"id": "1"}, "id": "call_1"}
    gen.message.tool_calls = [tc]
    response.generations = [[gen]]

    with reagent_flow.session("test") as s:
        handler.on_llm_end(response, run_id=uuid4())

    assert len(s.trace.turns) == 1
    assert s.trace.turns[0].llm_call.tool_calls[0].name == "lookup"


def test_handler_captures_tool_result() -> None:
    handler = ReagentCallbackHandler()

    response = MagicMock()
    gen = MagicMock()
    gen.text = None
    gen.message.tool_calls = [{"name": "lookup", "args": {}, "id": "c1"}]
    response.generations = [[gen]]

    with reagent_flow.session("test") as s:
        handler.on_llm_end(response, run_id=uuid4())
        handler.on_tool_end("result data", run_id=uuid4(), name="lookup")

    assert len(s.trace.turns[0].tool_results) == 1
    assert s.trace.turns[0].tool_results[0].result == "result data"


def test_handler_captures_tool_error() -> None:
    handler = ReagentCallbackHandler()

    response = MagicMock()
    gen = MagicMock()
    gen.text = None
    gen.message.tool_calls = [{"name": "lookup", "args": {}, "id": "c1"}]
    response.generations = [[gen]]

    with reagent_flow.session("test") as s:
        handler.on_llm_end(response, run_id=uuid4())
        handler.on_tool_error(ValueError("not found"), run_id=uuid4(), name="lookup")

    assert s.trace.turns[0].tool_results[0].error == "not found"


def test_handler_noop_without_session() -> None:
    handler = ReagentCallbackHandler()
    response = MagicMock()
    gen = MagicMock()
    gen.text = "hello"
    gen.message.tool_calls = []
    response.generations = [[gen]]
    handler.on_llm_end(response, run_id=uuid4())
    # Should not raise


def test_handler_out_of_order_tool_results() -> None:
    """P1 fix: Results arriving out of order should still correlate by name."""
    handler = ReagentCallbackHandler()

    response = MagicMock()
    gen = MagicMock()
    gen.text = None
    gen.message.tool_calls = [
        {"name": "alpha", "args": {"x": 1}, "id": "call_alpha"},
        {"name": "beta", "args": {"y": 2}, "id": "call_beta"},
    ]
    response.generations = [[gen]]

    with reagent_flow.session("test") as s:
        handler.on_llm_end(response, run_id=uuid4())
        # beta finishes BEFORE alpha (out of order)
        handler.on_tool_end("beta_result", run_id=uuid4(), name="beta")
        handler.on_tool_end("alpha_result", run_id=uuid4(), name="alpha")

    results = s.trace.turns[0].tool_results
    assert len(results) == 2
    # beta's result should have beta's call_id, not alpha's
    assert results[0].call_id == "call_beta"
    assert results[0].result == "beta_result"
    assert results[1].call_id == "call_alpha"
    assert results[1].result == "alpha_result"


def test_handler_parallel_same_name_tool_calls_in_order() -> None:
    """Same-name tools completing in order should correlate correctly."""
    handler = ReagentCallbackHandler()

    response = MagicMock()
    gen = MagicMock()
    gen.text = None
    gen.message.tool_calls = [
        {"name": "lookup", "args": {"id": "1"}, "id": "call_a"},
        {"name": "lookup", "args": {"id": "2"}, "id": "call_b"},
    ]
    response.generations = [[gen]]

    run_a, run_b = uuid4(), uuid4()

    with reagent_flow.session("test") as s:
        handler.on_llm_end(response, run_id=uuid4())
        # on_tool_start fires in request order
        handler.on_tool_start({"name": "lookup"}, "", run_id=run_a)
        handler.on_tool_start({"name": "lookup"}, "", run_id=run_b)
        # Completions arrive in order
        handler.on_tool_end("result_1", run_id=run_a, name="lookup")
        handler.on_tool_end("result_2", run_id=run_b, name="lookup")

    results = s.trace.turns[0].tool_results
    assert len(results) == 2
    assert results[0].call_id == "call_a"
    assert results[0].result == "result_1"
    assert results[1].call_id == "call_b"
    assert results[1].result == "result_2"


def test_handler_parallel_same_name_tool_calls_out_of_order() -> None:
    """Same-name tools completing out of order should still correlate correctly via run_id."""
    handler = ReagentCallbackHandler()

    response = MagicMock()
    gen = MagicMock()
    gen.text = None
    gen.message.tool_calls = [
        {"name": "lookup", "args": {"id": "1"}, "id": "call_a"},
        {"name": "lookup", "args": {"id": "2"}, "id": "call_b"},
    ]
    response.generations = [[gen]]

    run_a, run_b = uuid4(), uuid4()

    with reagent_flow.session("test") as s:
        handler.on_llm_end(response, run_id=uuid4())
        # on_tool_start fires in request order: a then b
        handler.on_tool_start({"name": "lookup"}, "", run_id=run_a)
        handler.on_tool_start({"name": "lookup"}, "", run_id=run_b)
        # Completions arrive REVERSED: b finishes before a
        handler.on_tool_end("result_b", run_id=run_b, name="lookup")
        handler.on_tool_end("result_a", run_id=run_a, name="lookup")

    results = s.trace.turns[0].tool_results
    assert len(results) == 2
    # result_b arrived first but should have call_b's id
    assert results[0].call_id == "call_b"
    assert results[0].result == "result_b"
    # result_a arrived second but should have call_a's id
    assert results[1].call_id == "call_a"
    assert results[1].result == "result_a"
