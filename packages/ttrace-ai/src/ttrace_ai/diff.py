"""Trace diff engine for golden baseline comparison."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ttrace_ai.models import Trace, Turn


@dataclass
class TurnDiff:
    """Diff result for a single turn."""

    turn_index: int
    tool_name_expected: str | None = None
    tool_name_actual: str | None = None
    tool_name_match: bool = True
    arguments_match: bool = True
    argument_diffs: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    result_match: bool = True
    extra: bool = False
    missing: bool = False

    @property
    def is_match(self) -> bool:
        """Return True if this turn matches the golden baseline."""
        return (
            self.tool_name_match
            and self.arguments_match
            and self.result_match
            and not self.extra
            and not self.missing
        )


@dataclass
class DiffResult:
    """Full diff result between two traces."""

    golden_name: str
    actual_name: str
    turn_diffs: list[TurnDiff] = field(default_factory=list)
    golden_turn_count: int = 0
    actual_turn_count: int = 0

    @property
    def is_match(self) -> bool:
        """Return True if all turns match."""
        return all(td.is_match for td in self.turn_diffs)

    def summary(self) -> str:
        """Return a human-readable summary of the diff."""
        if self.is_match:
            return f"Traces match ({self.golden_turn_count} turns)"
        lines: list[str] = []
        lines.append(
            f"Trace diff: {self.golden_turn_count} golden turns"
            f" vs {self.actual_turn_count} actual turns"
        )
        for td in self.turn_diffs:
            if not td.is_match:
                if td.missing:
                    lines.append(
                        f"  Turn {td.turn_index}: MISSING (expected {td.tool_name_expected})"
                    )
                elif td.extra:
                    lines.append(
                        f"  Turn {td.turn_index}: EXTRA (got {td.tool_name_actual})"
                    )
                elif not td.tool_name_match:
                    lines.append(
                        f"  Turn {td.turn_index}: tool mismatch "
                        f"(expected {td.tool_name_expected}, got {td.tool_name_actual})"
                    )
                elif not td.arguments_match:
                    lines.append(f"  Turn {td.turn_index}: argument differences:")
                    for key, (exp, act) in td.argument_diffs.items():
                        lines.append(f"    {key}: expected={exp!r}, actual={act!r}")
                elif not td.result_match:
                    lines.append(
                        f"  Turn {td.turn_index}: result mismatch"
                    )
        return "\n".join(lines)


def diff_traces(golden: Trace, actual: Trace) -> DiffResult:
    """Compare an actual trace against a golden baseline."""
    g_len = len(golden.turns)
    a_len = len(actual.turns)
    result = DiffResult(
        golden_name=golden.name,
        actual_name=actual.name,
        golden_turn_count=g_len,
        actual_turn_count=a_len,
    )

    for i in range(max(g_len, a_len)):
        g_turn = golden.turns[i] if i < g_len else None
        a_turn = actual.turns[i] if i < a_len else None

        if g_turn is None and a_turn is not None:
            g_name = _primary_tool(a_turn)
            result.turn_diffs.append(
                TurnDiff(turn_index=i, tool_name_actual=g_name, extra=True)
            )
            continue

        if a_turn is None and g_turn is not None:
            g_name = _primary_tool(g_turn)
            result.turn_diffs.append(
                TurnDiff(turn_index=i, tool_name_expected=g_name, missing=True)
            )
            continue

        assert g_turn is not None and a_turn is not None
        td = _diff_turn(i, g_turn, a_turn)
        result.turn_diffs.append(td)

    return result


def _primary_tool(turn: Turn) -> str | None:
    """Get the primary tool name from a turn."""
    if turn.llm_call.tool_calls:
        return turn.llm_call.tool_calls[0].name
    return None


def _diff_turn(index: int, golden: Turn, actual: Turn) -> TurnDiff:
    """Diff two turns."""
    td = TurnDiff(turn_index=index)

    g_tools = golden.llm_call.tool_calls
    a_tools = actual.llm_call.tool_calls

    g_name = g_tools[0].name if g_tools else None
    a_name = a_tools[0].name if a_tools else None
    td.tool_name_expected = g_name
    td.tool_name_actual = a_name

    if g_name != a_name:
        td.tool_name_match = False
        return td

    if g_tools and a_tools:
        g_args = g_tools[0].arguments
        a_args = a_tools[0].arguments
        all_keys = set(g_args.keys()) | set(a_args.keys())
        for key in all_keys:
            g_val = g_args.get(key)
            a_val = a_args.get(key)
            if g_val != a_val:
                td.arguments_match = False
                td.argument_diffs[key] = (g_val, a_val)

    # Compare tool results
    g_results = sorted(
        ((tr.call_id, tr.result, tr.error) for tr in golden.tool_results),
        key=lambda x: x[0],
    )
    a_results = sorted(
        ((tr.call_id, tr.result, tr.error) for tr in actual.tool_results),
        key=lambda x: x[0],
    )
    if len(g_results) != len(a_results):
        td.result_match = False
    else:
        for (_, g_res, g_err), (_, a_res, a_err) in zip(g_results, a_results):
            if g_res != a_res or g_err != a_err:
                td.result_match = False
                break

    # Compare response text
    g_text = golden.llm_call.response_text
    a_text = actual.llm_call.response_text
    if g_text != a_text:
        td.result_match = False

    return td
