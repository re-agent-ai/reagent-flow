"""Trace diff engine for golden baseline comparison."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from reagent_ai.models import Trace, Turn


@dataclass
class TurnDiff:
    """Diff result for a single turn."""

    turn_index: int
    tool_name_expected: str | None = None
    tool_name_actual: str | None = None
    tool_names_expected: list[str] = field(default_factory=list)
    tool_names_actual: list[str] = field(default_factory=list)
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
                    lines.append(f"  Turn {td.turn_index}: EXTRA (got {td.tool_name_actual})")
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
                    lines.append(f"  Turn {td.turn_index}: result mismatch")
        return "\n".join(lines)


def diff_traces(
    golden: Trace,
    actual: Trace,
    *,
    ignore_fields: set[str] | None = None,
) -> DiffResult:
    """Compare an actual trace against a golden baseline.

    Args:
        golden: The golden baseline trace.
        actual: The actual trace to compare.
        ignore_fields: Set of field paths to ignore during comparison.
            Supported values: ``"arguments"``, ``"results"``,
            ``"response_text"``, or specific argument keys like
            ``"tool_name.arg_key"``.

    """
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
            result.turn_diffs.append(TurnDiff(turn_index=i, tool_name_actual=g_name, extra=True))
            continue

        if a_turn is None and g_turn is not None:
            g_name = _primary_tool(g_turn)
            result.turn_diffs.append(
                TurnDiff(turn_index=i, tool_name_expected=g_name, missing=True)
            )
            continue

        if g_turn is None or a_turn is None:  # pragma: no cover
            continue
        td = _diff_turn(i, g_turn, a_turn, ignore_fields=ignore_fields or set())
        result.turn_diffs.append(td)

    return result


def _primary_tool(turn: Turn) -> str | None:
    """Get the primary tool name from a turn."""
    if turn.llm_call.tool_calls:
        return turn.llm_call.tool_calls[0].name
    return None


def _diff_turn(
    index: int,
    golden: Turn,
    actual: Turn,
    *,
    ignore_fields: set[str],
) -> TurnDiff:
    """Diff two turns, comparing all tool calls by position."""
    td = TurnDiff(turn_index=index)

    g_tools = golden.llm_call.tool_calls
    a_tools = actual.llm_call.tool_calls

    # Populate primary tool name for backward compat with summary output
    td.tool_name_expected = g_tools[0].name if g_tools else None
    td.tool_name_actual = a_tools[0].name if a_tools else None
    td.tool_names_expected = [tc.name for tc in g_tools]
    td.tool_names_actual = [tc.name for tc in a_tools]

    # Compare tool call count first
    if len(g_tools) != len(a_tools):
        td.tool_name_match = False
        return td

    # Compare each tool call by position
    skip_all_args = "arguments" in ignore_fields
    for g_tc, a_tc in zip(g_tools, a_tools):
        if g_tc.name != a_tc.name:
            td.tool_name_match = False

        if not skip_all_args:
            all_keys = set(g_tc.arguments.keys()) | set(a_tc.arguments.keys())
            for key in all_keys:
                field_key = f"{g_tc.name}.{key}"
                if field_key in ignore_fields:
                    continue
                g_val = g_tc.arguments.get(key)
                a_val = a_tc.arguments.get(key)
                if g_val != a_val:
                    td.arguments_match = False
                    td.argument_diffs[field_key] = (g_val, a_val)

    # Compare tool results by position (not by random call_id)
    if "results" not in ignore_fields:
        g_results = [(tr.result, tr.error) for tr in golden.tool_results]
        a_results = [(tr.result, tr.error) for tr in actual.tool_results]
        if len(g_results) != len(a_results):
            td.result_match = False
        else:
            for (g_res, g_err), (a_res, a_err) in zip(g_results, a_results):
                if g_res != a_res or g_err != a_err:
                    td.result_match = False
                    break

    # Compare response text
    if "response_text" not in ignore_fields:
        g_text = golden.llm_call.response_text
        a_text = actual.llm_call.response_text
        if g_text != a_text:
            td.result_match = False

    return td
