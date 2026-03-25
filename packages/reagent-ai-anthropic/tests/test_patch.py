"""Tests for Anthropic adapter patch function."""

import warnings
from unittest.mock import MagicMock

import reagent_ai
from reagent_ai.exceptions import ReagentAdapterWarning
from reagent_ai_anthropic import patch


def _make_tool_use_block(name: str, input_data: dict, block_id: str = "toolu_01") -> MagicMock:
    """Create a mock Anthropic tool_use content block."""
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.input = input_data
    block.id = block_id
    return block


def _make_text_block(text: str) -> MagicMock:
    """Create a mock Anthropic text content block."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _mock_anthropic_client(
    tool_blocks: list | None = None,
    text: str | None = None,
) -> MagicMock:
    """Create a mock Anthropic client with a canned response."""
    client = MagicMock()
    response = MagicMock()
    response.model = "claude-sonnet-4-20250514"

    content = []
    if text:
        content.append(_make_text_block(text))
    if tool_blocks:
        content.extend(tool_blocks)

    response.content = content

    usage = MagicMock()
    usage.input_tokens = 100
    usage.output_tokens = 50
    response.usage = usage

    client.messages.create = MagicMock(return_value=response)
    return client


def test_patch_returns_client() -> None:
    """patch() should return the client object."""
    client = _mock_anthropic_client(text="Hello")
    result = patch(client)
    assert result is client


def test_patch_captures_tool_calls() -> None:
    """patch() should capture tool_use blocks as tool calls."""
    tool_block = _make_tool_use_block("search", {"query": "test"}, "toolu_abc")
    client = _mock_anthropic_client(tool_blocks=[tool_block])
    patched = patch(client)

    with reagent_ai.session("test") as s:
        patched.messages.create(model="claude-sonnet-4-20250514", messages=[], max_tokens=100)

    assert len(s.trace.turns) == 1
    tc = s.trace.turns[0].llm_call.tool_calls[0]
    assert tc.name == "search"
    assert tc.arguments == {"query": "test"}
    assert tc.call_id == "toolu_abc"


def test_patch_captures_text_response() -> None:
    """patch() should capture text content blocks."""
    client = _mock_anthropic_client(text="Hello, world!")
    patched = patch(client)

    with reagent_ai.session("test") as s:
        patched.messages.create(model="claude-sonnet-4-20250514", messages=[], max_tokens=100)

    assert s.trace.turns[0].llm_call.response_text == "Hello, world!"


def test_patch_captures_mixed_response() -> None:
    """patch() should capture both text and tool_use blocks."""
    tool_block = _make_tool_use_block("lookup", {"id": "123"})
    client = _mock_anthropic_client(tool_blocks=[tool_block], text="Let me look that up.")
    patched = patch(client)

    with reagent_ai.session("test") as s:
        patched.messages.create(model="claude-sonnet-4-20250514", messages=[], max_tokens=100)

    turn = s.trace.turns[0]
    assert turn.llm_call.response_text == "Let me look that up."
    assert len(turn.llm_call.tool_calls) == 1
    assert turn.llm_call.tool_calls[0].name == "lookup"


def test_patch_captures_token_usage() -> None:
    """patch() should extract input_tokens and output_tokens."""
    client = _mock_anthropic_client(text="Hi")
    patched = patch(client)

    with reagent_ai.session("test") as s:
        patched.messages.create(model="claude-sonnet-4-20250514", messages=[], max_tokens=100)

    usage = s.trace.turns[0].llm_call.token_usage
    assert usage == {"input_tokens": 100, "output_tokens": 50}


def test_patch_noop_without_session() -> None:
    """patch() should not raise when no session is active."""
    client = _mock_anthropic_client(text="Hi")
    patched = patch(client)
    result = patched.messages.create(model="claude-sonnet-4-20250514", messages=[], max_tokens=100)
    assert result is not None


def test_patch_warns_on_streaming() -> None:
    """Streaming calls should emit a warning and skip capture."""
    client = _mock_anthropic_client(text="Hi")
    patched = patch(client)
    with reagent_ai.session("test") as s:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            patched.messages.create(
                model="claude-sonnet-4-20250514",
                messages=[],
                max_tokens=100,
                stream=True,
            )
    assert len(s.trace.turns) == 0
    assert any(issubclass(w.category, ReagentAdapterWarning) for w in caught)


def test_patch_multiple_tool_use_blocks() -> None:
    """patch() should capture multiple parallel tool_use blocks."""
    blocks = [
        _make_tool_use_block("search", {"q": "a"}, "toolu_1"),
        _make_tool_use_block("lookup", {"id": "b"}, "toolu_2"),
    ]
    client = _mock_anthropic_client(tool_blocks=blocks)
    patched = patch(client)

    with reagent_ai.session("test") as s:
        patched.messages.create(model="claude-sonnet-4-20250514", messages=[], max_tokens=100)

    tcs = s.trace.turns[0].llm_call.tool_calls
    assert len(tcs) == 2
    assert tcs[0].name == "search"
    assert tcs[1].name == "lookup"
