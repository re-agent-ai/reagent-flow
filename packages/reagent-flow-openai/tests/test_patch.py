"""Tests for OpenAI adapter patch function."""

import warnings
from unittest.mock import MagicMock

import reagent_flow
from reagent_flow.exceptions import ReagentAdapterWarning
from reagent_flow_openai import patch


def _mock_openai_client() -> MagicMock:
    """Create a mock OpenAI client with chat.completions.create."""
    client = MagicMock()
    response = MagicMock()
    response.choices = [MagicMock()]
    func_mock = MagicMock()
    func_mock.name = "lookup_order"
    func_mock.arguments = '{"id": "123"}'
    tc_mock = MagicMock(id="call_1", function=func_mock)
    response.choices[0].message.tool_calls = [tc_mock]
    response.choices[0].message.content = None
    response.model = "gpt-4o"
    response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
    client.chat.completions.create.return_value = response
    return client


def test_patch_returns_client() -> None:
    client = _mock_openai_client()
    patched = patch(client)
    assert patched is client


def test_patch_captures_tool_calls() -> None:
    client = _mock_openai_client()
    patched = patch(client)
    with reagent_flow.session("test") as s:
        patched.chat.completions.create(model="gpt-4o", messages=[], tools=[])
    assert len(s.trace.turns) == 1
    assert s.trace.turns[0].llm_call.tool_calls[0].name == "lookup_order"


def test_patch_noop_without_session() -> None:
    client = _mock_openai_client()
    patched = patch(client)
    result = patched.chat.completions.create(model="gpt-4o", messages=[])
    assert result is not None


def test_patch_warns_on_streaming() -> None:
    """Streaming calls should emit a warning and skip capture."""
    client = _mock_openai_client()
    patched = patch(client)
    with reagent_flow.session("test") as s:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            patched.chat.completions.create(model="gpt-4o", messages=[], stream=True)
    assert len(s.trace.turns) == 0
    assert any(issubclass(w.category, ReagentAdapterWarning) for w in caught)


def _tool_call_response(name: str, arguments_json: str, call_id: str) -> MagicMock:
    """Build a canned OpenAI response carrying a single tool call."""
    response = MagicMock()
    response.choices = [MagicMock()]
    func_mock = MagicMock()
    func_mock.name = name
    func_mock.arguments = arguments_json
    tc_mock = MagicMock(id=call_id, function=func_mock)
    response.choices[0].message.tool_calls = [tc_mock]
    response.choices[0].message.content = None
    response.model = "gpt-4o"
    response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
    return response


def _text_response(text: str) -> MagicMock:
    """Build a canned OpenAI response with only a text message."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.tool_calls = None
    response.choices[0].message.content = text
    response.model = "gpt-4o"
    response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
    return response


def test_patch_captures_tool_result_from_followup_messages() -> None:
    """Tool results sent back in the next create() call are logged to the prior turn.

    Mirrors the real OpenAI tool-calling loop: turn 1 returns a tool_use,
    the caller runs the tool, then turn 2 is a create() with a
    ``{"role": "tool", ...}`` message carrying the result. The adapter
    should attach that result to turn 0 so ``assert_tool_output_matches``
    has data to validate.
    """
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _tool_call_response("lookup_order", '{"id": "A-1"}', "call_xyz"),
        _text_response("Order A-1 is shipped."),
    ]
    patched = patch(client)

    with reagent_flow.session("openai-toolresult") as s:
        patched.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
        )
        patched.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "call_xyz"}]},
                {
                    "role": "tool",
                    "tool_call_id": "call_xyz",
                    "content": '{"status": "shipped", "eta_days": 2}',
                },
            ],
        )

    assert len(s.trace.turns) == 2
    turn0 = s.trace.turns[0]
    assert len(turn0.tool_results) == 1
    tr = turn0.tool_results[0]
    assert tr.call_id == "call_xyz"
    assert tr.result == {"status": "shipped", "eta_days": 2}
    # Final-answer turn exists and has no trailing tool result attached.
    assert s.trace.turns[1].tool_results == []


def test_patch_tool_result_assert_tool_output_matches() -> None:
    """The captured tool result should satisfy assert_tool_output_matches."""
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _tool_call_response("extract_vendor_packet", '{"request_id": "VR-42"}', "call_1"),
        _text_response("done"),
    ]
    patched = patch(client)

    with reagent_flow.session("openai-schema") as s:
        patched.chat.completions.create(model="gpt-4o", messages=[])
        patched.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": '{"vendor_name": "ClearVoice AI", "contains_customer_pii": true}',
                },
            ],
        )

    s.assert_tool_output_matches(
        "extract_vendor_packet",
        schema={"vendor_name": str, "contains_customer_pii": bool},
    )


def test_patch_tool_result_plain_string_content() -> None:
    """Non-JSON string content is preserved verbatim as the tool result."""
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _tool_call_response("ping", "{}", "call_ping"),
        _text_response("ok"),
    ]
    patched = patch(client)

    with reagent_flow.session("openai-plain") as s:
        patched.chat.completions.create(model="gpt-4o", messages=[])
        patched.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "tool", "tool_call_id": "call_ping", "content": "pong"}],
        )

    assert s.trace.turns[0].tool_results[0].result == "pong"


def test_patch_tool_result_unknown_call_id_ignored() -> None:
    """Tool messages referencing an unknown call_id are silently skipped."""
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _tool_call_response("lookup", "{}", "call_real"),
        _text_response("ok"),
    ]
    patched = patch(client)

    with reagent_flow.session("openai-unknown") as s:
        patched.chat.completions.create(model="gpt-4o", messages=[])
        patched.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "tool", "tool_call_id": "call_ghost", "content": "x"}],
        )

    # Only the real call exists; no ghost result was attached.
    assert s.trace.turns[0].tool_results == []
