"""Example: testing an agent with explicit logging (no adapter needed)."""

import ttrace_ai


def fake_agent_run(session: ttrace_ai.Session) -> None:
    """Simulate an agent that looks up an order and processes a refund."""
    session.log_llm_call(
        tool_calls=[{"name": "lookup_order", "arguments": {"order_id": "123"}}],
    )
    session.log_tool_result("lookup_order", result={"status": "active", "amount": 49.99})

    session.log_llm_call(
        tool_calls=[
            {"name": "process_refund", "arguments": {"order_id": "123", "amount": 49.99}},
        ],
    )
    session.log_tool_result("process_refund", result={"success": True, "refund_id": "R456"})

    session.log_llm_call(response_text="Refund processed successfully.", tool_calls=[])


def test_refund_flow(tmp_path: object) -> None:
    with ttrace_ai.session("refund_flow", trace_dir=str(tmp_path)) as s:
        fake_agent_run(s)

    s.assert_called("lookup_order")
    s.assert_called_before("lookup_order", "process_refund")
    s.assert_tool_succeeded("process_refund")
    s.assert_never_called("delete_account")
    s.assert_max_turns(5)
