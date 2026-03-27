"""Tests for reagent_flow.models."""

from reagent_flow.models import LLMCall, Message, ToolCall, ToolResult, Trace, Turn


def test_tool_call_creation() -> None:
    tc = ToolCall(name="lookup", arguments={"id": "1"}, call_id="abc", timestamp=1.0)
    assert tc.name == "lookup"
    assert tc.arguments == {"id": "1"}
    assert tc.call_id == "abc"
    assert tc.timestamp == 1.0


def test_tool_result_defaults() -> None:
    tr = ToolResult(call_id="abc", result={"ok": True})
    assert tr.error is None
    assert tr.duration_ms == 0


def test_message_str_content() -> None:
    m = Message(role="user", content="hello")
    assert m.role == "user"
    assert m.content == "hello"


def test_message_list_content() -> None:
    m = Message(role="user", content=[{"type": "text", "text": "hi"}])
    assert isinstance(m.content, list)


def test_llm_call_minimal() -> None:
    lc = LLMCall(messages=None, response_text="answer", tool_calls=[])
    assert lc.messages is None
    assert lc.model is None
    assert lc.token_usage is None
    assert lc.timestamp == 0


def test_turn_creation() -> None:
    lc = LLMCall(messages=None, response_text=None, tool_calls=[])
    t = Turn(index=0, llm_call=lc, tool_results=[])
    assert t.index == 0
    assert t.duration_ms == 0


def test_trace_defaults() -> None:
    tr = Trace(trace_id="id1", name="test")
    assert tr.turns == []
    assert tr.metadata == {}
    assert tr.ended_at is None
    assert tr.format_version == "1"


def test_trace_serialization_roundtrip() -> None:
    tc = ToolCall(name="lookup", arguments={"id": "1"}, call_id="c1", timestamp=1.0)
    tr_result = ToolResult(call_id="c1", result={"found": True}, duration_ms=50.0)
    msg = Message(role="user", content="test")
    lc = LLMCall(messages=[msg], response_text=None, tool_calls=[tc], model="gpt-4o", timestamp=1.0)
    turn = Turn(index=0, llm_call=lc, tool_results=[tr_result], duration_ms=100.0)
    trace = Trace(trace_id="t1", name="test_roundtrip", turns=[turn], started_at=1.0, ended_at=2.0)

    from reagent_flow.models import trace_from_dict, trace_to_dict

    d = trace_to_dict(trace)
    restored = trace_from_dict(d)
    assert restored.trace_id == trace.trace_id
    assert restored.name == trace.name
    assert len(restored.turns) == 1
    assert restored.turns[0].llm_call.tool_calls[0].name == "lookup"
    assert restored.turns[0].tool_results[0].result == {"found": True}
    assert restored.format_version == "1"
