"""Tests for LangGraph adapter."""

from unittest.mock import MagicMock
from uuid import uuid4

import reagent_flow
from reagent_flow_langgraph import ReagentGraphTracer


def test_tracer_inherits_langchain_capture() -> None:
    tracer = ReagentGraphTracer()

    response = MagicMock()
    gen = MagicMock()
    gen.text = "answer"
    gen.message.tool_calls = []
    response.generations = [[gen]]

    with reagent_flow.session("test") as s:
        tracer.on_llm_end(response, run_id=uuid4())

    assert len(s.trace.turns) == 1
    assert s.trace.turns[0].llm_call.response_text == "answer"


def test_tracer_captures_tool_calls_with_chain_context() -> None:
    tracer = ReagentGraphTracer()

    response = MagicMock()
    gen = MagicMock()
    gen.text = None
    tc = {"name": "search", "args": {"q": "test"}, "id": "c1"}
    gen.message.tool_calls = [tc]
    response.generations = [[gen]]

    with reagent_flow.session("test") as s:
        tracer.on_chain_start({"name": "agent_node"}, run_id=uuid4())
        tracer.on_llm_end(response, run_id=uuid4())
        tracer.on_tool_end("results", run_id=uuid4(), name="search")

    assert len(s.trace.turns) == 1
    assert s.trace.turns[0].llm_call.tool_calls[0].name == "search"
    assert tracer._current_node == "agent_node"


def test_tracer_noop_without_session() -> None:
    tracer = ReagentGraphTracer()
    response = MagicMock()
    gen = MagicMock()
    gen.text = "hello"
    gen.message.tool_calls = []
    response.generations = [[gen]]
    tracer.on_llm_end(response, run_id=uuid4())
    # Should not raise
