"""Tests for OpenAI adapter patch function."""

from unittest.mock import MagicMock

import ttrace_ai
from ttrace_ai_openai import patch


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
    with ttrace_ai.session("test") as s:
        patched.chat.completions.create(model="gpt-4o", messages=[], tools=[])
    assert len(s.trace.turns) == 1
    assert s.trace.turns[0].llm_call.tool_calls[0].name == "lookup_order"


def test_patch_noop_without_session() -> None:
    client = _mock_openai_client()
    patched = patch(client)
    result = patched.chat.completions.create(model="gpt-4o", messages=[])
    assert result is not None
