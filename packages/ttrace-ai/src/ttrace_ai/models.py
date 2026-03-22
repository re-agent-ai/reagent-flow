"""Data models for ttrace-ai traces."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

FORMAT_VERSION = "1"


@dataclass
class ToolCall:
    """A tool call requested by the LLM."""

    name: str
    arguments: dict[str, Any]
    call_id: str
    timestamp: float


@dataclass
class ToolResult:
    """Result of executing a tool call."""

    call_id: str
    result: Any
    error: str | None = None
    duration_ms: float = 0


@dataclass
class Message:
    """A message in the conversation."""

    role: str
    content: str | list[Any]


@dataclass
class LLMCall:
    """An LLM call including its response."""

    messages: list[Message] | None
    response_text: str | None
    tool_calls: list[ToolCall]
    model: str | None = None
    token_usage: dict[str, Any] | None = None
    timestamp: float = 0


@dataclass
class Turn:
    """A single turn in an agent trace."""

    index: int
    llm_call: LLMCall
    tool_results: list[ToolResult]
    duration_ms: float = 0


@dataclass
class Trace:
    """A complete agent trace."""

    trace_id: str
    name: str
    turns: list[Turn] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at: float = 0
    ended_at: float | None = None
    format_version: str = FORMAT_VERSION


def trace_to_dict(trace: Trace) -> dict[str, Any]:
    """Serialize a Trace to a dict."""
    return asdict(trace)


def trace_from_dict(d: dict[str, Any]) -> Trace:
    """Deserialize a Trace from a dict."""
    turns: list[Turn] = []
    for t in d.get("turns", []):
        lc_data = t["llm_call"]
        messages: list[Message] | None = None
        if lc_data.get("messages"):
            messages = [Message(**m) for m in lc_data["messages"]]
        tool_calls = [ToolCall(**tc) for tc in lc_data.get("tool_calls", [])]
        llm_call = LLMCall(
            messages=messages,
            response_text=lc_data.get("response_text"),
            tool_calls=tool_calls,
            model=lc_data.get("model"),
            token_usage=lc_data.get("token_usage"),
            timestamp=lc_data.get("timestamp", 0),
        )
        tool_results = [ToolResult(**tr) for tr in t.get("tool_results", [])]
        turns.append(
            Turn(
                index=t["index"],
                llm_call=llm_call,
                tool_results=tool_results,
                duration_ms=t.get("duration_ms", 0),
            )
        )
    return Trace(
        trace_id=d["trace_id"],
        name=d["name"],
        turns=turns,
        metadata=d.get("metadata", {}),
        started_at=d.get("started_at", 0),
        ended_at=d.get("ended_at"),
        format_version=d.get("format_version", FORMAT_VERSION),
    )
