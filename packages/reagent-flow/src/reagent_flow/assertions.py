"""Assertion implementations for reagent-flow sessions."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from types import EllipsisType
from typing import TYPE_CHECKING, Any

from reagent_flow.exceptions import ReagentWarning
from reagent_flow.stacktrace import format_stack_trace

if TYPE_CHECKING:
    from reagent_flow.models import Trace


def _assertion_error(trace: Trace, msg: str, expected_tool: str | None = None) -> AssertionError:
    """Create an AssertionError with Agent Stack Trace attached."""
    stack = format_stack_trace(trace, assertion_msg=msg, expected_tool=expected_tool, color=True)
    return AssertionError(f"\n{stack}")


def _all_tool_names(trace: Trace) -> set[str]:
    """Return set of all tool names called in the trace."""
    return set(_flatten_tool_names(trace))


def _tool_call_count(trace: Trace) -> int:
    """Return total number of tool calls."""
    return sum(len(turn.llm_call.tool_calls) for turn in trace.turns)


def _flatten_tool_names(trace: Trace) -> list[str]:
    """Flatten all tool call names from a trace into an ordered list.

    Iterates turns in index order. Within each turn, iterates tool_calls
    in list order. Text-only turns (no tool calls) are skipped.
    """
    names: list[str] = []
    for turn in trace.turns:
        for tc in turn.llm_call.tool_calls:
            names.append(tc.name)
    return names


def assert_called(trace: Trace, tool_name: str) -> None:
    """Assert that a tool was called at least once."""
    if tool_name not in _all_tool_names(trace):
        total = _tool_call_count(trace)
        raise _assertion_error(
            trace,
            f'"{tool_name}" was never called ({len(trace.turns)} turns, {total} tool calls)',
            expected_tool=tool_name,
        )


def assert_never_called(trace: Trace, tool_name: str) -> None:
    """Assert that a tool was never called."""
    if tool_name not in _all_tool_names(trace):
        return
    indices = [
        turn.index
        for turn in trace.turns
        for tc in turn.llm_call.tool_calls
        if tc.name == tool_name
    ]
    raise _assertion_error(
        trace,
        f'"{tool_name}" was called in turn(s) {indices}',
    )


def assert_called_before(trace: Trace, first: str, second: str) -> None:
    """Assert that first tool was called before second tool.

    Uses positional ordering across the flattened tool call list so that
    parallel calls within the same turn are compared by list position.
    """
    first_pos: int | None = None
    second_pos: int | None = None
    pos = 0
    for turn in trace.turns:
        for tc in turn.llm_call.tool_calls:
            if tc.name == first and first_pos is None:
                first_pos = pos
            if tc.name == second and second_pos is None:
                second_pos = pos
            pos += 1
    if first_pos is None:
        raise _assertion_error(trace, f'"{first}" was never called')
    if second_pos is None:
        raise _assertion_error(trace, f'"{second}" was never called')
    if first_pos >= second_pos:
        raise _assertion_error(
            trace,
            f'"{first}" (position {first_pos}) was not called before '
            f'"{second}" (position {second_pos})',
        )


def assert_tool_succeeded(trace: Trace, tool_name: str) -> None:
    """Assert that a tool was called and all its executions succeeded."""
    found = False
    for turn in trace.turns:
        for tc in turn.llm_call.tool_calls:
            if tc.name == tool_name:
                found = True
                matching_results = [tr for tr in turn.tool_results if tr.call_id == tc.call_id]
                if not matching_results:
                    raise _assertion_error(
                        trace,
                        f'"{tool_name}" in turn {turn.index} has no recorded result',
                        expected_tool=tool_name,
                    )
                for tr in matching_results:
                    if tr.error is not None:
                        raise _assertion_error(
                            trace,
                            f'"{tool_name}" failed in turn {turn.index}: {tr.error}',
                            expected_tool=tool_name,
                        )
    if not found:
        raise _assertion_error(
            trace,
            f'"{tool_name}" was never called ({len(trace.turns)} turns, '
            f"{_tool_call_count(trace)} tool calls)",
            expected_tool=tool_name,
        )


def assert_max_turns(trace: Trace, n: int) -> None:
    """Assert that the trace has at most n turns."""
    if len(trace.turns) > n:
        raise _assertion_error(trace, f"Expected at most {n} turns, got {len(trace.turns)}")


def assert_total_duration_under(trace: Trace, *, ms: float) -> None:
    """Assert that total trace duration is under ms milliseconds.

    Works inside active sessions by using the current wall clock time
    when ``ended_at`` has not been set yet.
    """
    import time

    end = trace.ended_at if trace.ended_at is not None else time.time()
    actual_ms = (end - trace.started_at) * 1000
    if actual_ms > ms:
        raise _assertion_error(
            trace,
            f"Total duration {actual_ms:.0f}ms exceeds limit of {ms:.0f}ms",
        )


def _fmt_pattern(pattern: list[str | EllipsisType]) -> str:
    """Format a flow pattern for error messages."""
    parts: list[str] = []
    for elem in pattern:
        if elem is ...:
            parts.append("...")
        else:
            parts.append(repr(elem))
    return "[" + ", ".join(parts) + "]"


def assert_flow(trace: Trace, pattern: list[str | EllipsisType]) -> None:
    """Assert that tool calls match a flow pattern.

    Patterns are anchored to start and end by default. Use ``...`` (Ellipsis)
    to unanchor or allow gaps between elements.

    Examples::

        assert_flow(trace, ["search", "summarize"])         # exact consecutive match
        assert_flow(trace, ["search", ..., "summarize"])    # gap allowed
        assert_flow(trace, [..., "search", ..., "summarize", ...])  # anywhere in trace
    """
    flat = _flatten_tool_names(trace)

    # Validate: only str and Ellipsis are allowed
    for i, elem in enumerate(pattern):
        if elem is not ... and not isinstance(elem, str):
            msg = (
                f"Invalid pattern element at index {i}: {elem!r} "
                f"(type {type(elem).__name__}). Only str and ... are allowed."
            )
            raise TypeError(msg)

    # Normalize: collapse consecutive Ellipsis
    normalized: list[str | EllipsisType] = []
    for elem in pattern:
        if elem is ... and normalized and normalized[-1] is ...:
            continue
        normalized.append(elem)

    # Extract string elements and determine anchoring
    if not any(isinstance(e, str) for e in normalized):
        return  # empty or all-ellipsis pattern always passes

    has_leading_ellipsis = bool(normalized and normalized[0] is ...)
    has_trailing_ellipsis = bool(normalized and normalized[-1] is ...)

    # Build segments: groups of adjacent strings separated by ellipsis
    segments: list[list[str]] = []
    current_segment: list[str] = []

    for elem in normalized:
        if elem is ...:
            if current_segment:
                segments.append(current_segment)
                current_segment = []
        else:
            current_segment.append(elem)

    if current_segment:
        segments.append(current_segment)

    # Determine which segments can search vs must match at cursor position
    segment_modes: list[str] = []
    for i, _seg in enumerate(segments):
        if i == 0 and not has_leading_ellipsis:
            segment_modes.append("anchored")
        else:
            segment_modes.append("search")

    # Match segments against flat list
    cursor = 0
    for segment, mode in zip(segments, segment_modes):
        if mode == "anchored":
            for j, name in enumerate(segment):
                pos = cursor + j
                if pos >= len(flat) or flat[pos] != name:
                    raise _assertion_error(
                        trace,
                        f"Flow mismatch: expected '{name}' at position {pos}, "
                        f"got '{flat[pos] if pos < len(flat) else '<end>'}'. "
                        f"Pattern: {_fmt_pattern(pattern)}, actual: {flat}",
                    )
            cursor += len(segment)
        else:
            found = False
            for start in range(cursor, len(flat) - len(segment) + 1):
                if all(flat[start + j] == segment[j] for j in range(len(segment))):
                    cursor = start + len(segment)
                    found = True
                    break
            if not found:
                raise _assertion_error(
                    trace,
                    f"Flow mismatch: could not find {segment} after position {cursor}. "
                    f"Pattern: {_fmt_pattern(pattern)}, actual: {flat}",
                )

    # Check trailing anchor
    if not has_trailing_ellipsis and cursor != len(flat):
        raise _assertion_error(
            trace,
            f"Flow mismatch: unexpected trailing calls {flat[cursor:]}. "
            f"Pattern: {_fmt_pattern(pattern)}, actual: {flat}",
        )


def assert_called_times(
    trace: Trace, tool_name: str, *, min: int = 0, max: int | None = None
) -> None:
    """Assert that a tool was called between min and max times."""
    if max is not None and max < min:
        msg = f"max ({max}) is less than min ({min})"
        raise ValueError(msg)

    count = sum(
        1 for turn in trace.turns for tc in turn.llm_call.tool_calls if tc.name == tool_name
    )
    if count < min:
        raise _assertion_error(
            trace,
            f'"{tool_name}" called {count} times, expected at least {min}',
            expected_tool=tool_name,
        )
    if max is not None and count > max:
        raise _assertion_error(
            trace,
            f'"{tool_name}" called {count} times, expected at most {max}',
            expected_tool=tool_name,
        )


def assert_handoff_received(child_trace: Trace, parent_trace: Trace) -> None:
    """Assert that a child trace is linked to a parent trace."""
    if child_trace.parent_trace_id is None:
        raise _assertion_error(
            child_trace,
            f'"{child_trace.name}" has no parent_trace_id set',
        )
    if child_trace.parent_trace_id != parent_trace.trace_id:
        raise _assertion_error(
            child_trace,
            f'"{child_trace.name}" parent_trace_id "{child_trace.parent_trace_id}" '
            f'does not match expected parent "{parent_trace.trace_id}"',
        )


def _require_handoff_context(child_trace: Trace) -> dict[str, Any]:
    """Validate and return handoff_context as a dict, or raise."""
    if child_trace.handoff_context is None:
        raise _assertion_error(
            child_trace,
            f'"{child_trace.name}" has no handoff_context set',
        )
    if not isinstance(child_trace.handoff_context, dict):
        raise _assertion_error(
            child_trace,
            f'"{child_trace.name}" handoff_context must be a dict, '
            f"got {type(child_trace.handoff_context).__name__}",
        )
    return child_trace.handoff_context


def assert_handoff_has_fields(child_trace: Trace, *, fields: list[str]) -> None:
    """Assert that required fields exist and are non-None in handoff context."""
    ctx = _require_handoff_context(child_trace)
    missing: list[str] = []
    for f in fields:
        if f not in ctx or ctx[f] is None:
            missing.append(f)
    if missing:
        raise _assertion_error(
            child_trace,
            f'"{child_trace.name}" handoff_context missing or None for field(s): {missing}',
        )


def _strict_isinstance(value: Any, expected_type: type) -> bool:
    """Like isinstance() but bool and int are treated as distinct types.

    Python's bool subclasses int, so isinstance(True, int) is True.
    For contract testing, we want strict type separation:
    - int schema rejects bool values
    - bool schema rejects int values
    """
    if expected_type is int and isinstance(value, bool):
        return False
    if expected_type is bool and not isinstance(value, bool):
        return False
    return isinstance(value, expected_type)


SchemaValue = type | dict[str, Any] | list[Any]
Schema = Mapping[str, SchemaValue]


def _validate_schema(
    value: Any,
    schema: SchemaValue,
    path: str,
    trace: Trace,
    context: str,
) -> None:
    """Recursively validate a value against a schema.

    Args:
        value: The actual value to validate.
        schema: A type (isinstance check), dict (nested schema), or list (typed list).
        path: Dot/bracket-notation path for error messages.
        trace: The trace for error formatting.
        context: Prefix for error messages (e.g. "handoff field" or "tool 'search' result field").

    """
    # Case 1: schema is a Python type — strict isinstance check (bool≠int)
    if isinstance(schema, type):
        if not _strict_isinstance(value, schema):
            raise _assertion_error(
                trace,
                f"{context} '{path}': expected {schema.__name__}, got {type(value).__name__}",
            )
        return

    # Case 2: schema is a dict — nested dict validation
    if isinstance(schema, dict):
        if not isinstance(value, dict):
            raise _assertion_error(
                trace,
                f"{context} '{path}': expected dict, got {type(value).__name__}",
            )
        for key, sub_schema in schema.items():
            child_path = f"{path}.{key}" if path else key
            if key not in value:
                raise _assertion_error(
                    trace,
                    f"{context} '{child_path}': missing from data",
                )
            _validate_schema(value[key], sub_schema, child_path, trace, context)
        return

    # Case 3: schema is a list — typed list validation
    if isinstance(schema, list):
        if not isinstance(value, list):
            raise _assertion_error(
                trace,
                f"{context} '{path}': expected list, got {type(value).__name__}",
            )
        if not schema:
            return  # empty schema list = no element constraints

        # Check if it's a list-of-dicts schema: [{"key": type}]
        if len(schema) == 1 and isinstance(schema[0], dict):
            dict_schema = schema[0]
            for i, elem in enumerate(value):
                elem_path = f"{path}[{i}]"
                _validate_schema(elem, dict_schema, elem_path, trace, context)
            return

        # Otherwise it's a union type list: [str], [str, int], etc.
        # All elements in schema must be types
        for i, elem in enumerate(value):
            if not any(_strict_isinstance(elem, t) for t in schema):
                type_names = " | ".join(t.__name__ for t in schema)
                raise _assertion_error(
                    trace,
                    f"{context} '{path}[{i}]': expected {type_names}, got {type(elem).__name__}",
                )
        return

    msg = f"Invalid schema value at '{path}': {schema!r}"
    raise TypeError(msg)


def assert_handoff_matches(child_trace: Trace, *, schema: Schema) -> None:
    """Validate handoff_context against a schema.

    Supports flat types (v0.3), nested dicts, typed lists, and
    list-of-dicts schemas (v0.4). Also accepts a Pydantic BaseModel class.
    """
    ctx = _require_handoff_context(child_trace)

    # Pydantic model detection (runtime only, no import at module level)
    if isinstance(schema, type) and hasattr(schema, "model_validate"):
        try:
            schema.model_validate(ctx)
        except Exception as e:
            raise _assertion_error(
                child_trace,
                f"handoff context validation failed: {e}",
            ) from None
        return

    for field, field_schema in schema.items():
        if field not in ctx:
            raise _assertion_error(
                child_trace,
                f"handoff field '{field}': missing from handoff_context",
            )
        _validate_schema(
            ctx[field],
            field_schema,
            field,
            child_trace,
            "handoff field",
        )


def assert_no_extra_fields(child_trace: Trace, *, allowed: list[str]) -> None:
    """Fail if handoff_context contains any key not in allowed."""
    ctx = _require_handoff_context(child_trace)
    allowed_set = set(allowed)
    extras = set(ctx.keys()) - allowed_set
    if extras:
        raise _assertion_error(
            child_trace,
            f"unexpected handoff fields: {extras}",
        )


def assert_tool_output_matches(
    trace: Trace,
    tool_name: str,
    *,
    schema: Schema,
) -> None:
    """Validate tool result values against a schema.

    Supports flat types (v0.3), nested dicts, typed lists, and
    list-of-dicts schemas (v0.4).
    """
    matched_results: list[tuple[str, Any]] = []

    for turn in trace.turns:
        for tc in turn.llm_call.tool_calls:
            if tc.name != tool_name:
                continue
            for tr in turn.tool_results:
                if tr.call_id == tc.call_id:
                    if tr.error is not None:
                        continue
                    matched_results.append((tr.call_id, tr.result))

    if not matched_results:
        raise _assertion_error(
            trace,
            f'"{tool_name}" was never called or all calls errored',
            expected_tool=tool_name,
        )

    # Validate all results are dicts before schema checking
    for call_id, result in matched_results:
        if not isinstance(result, dict):
            raise _assertion_error(
                trace,
                f"tool '{tool_name}' result is not a dict, got "
                f"{type(result).__name__} (call_id: {call_id})",
                expected_tool=tool_name,
            )

    # Pydantic model detection (runtime only, no import at module level)
    if isinstance(schema, type) and hasattr(schema, "model_validate"):
        for call_id, result in matched_results:
            try:
                schema.model_validate(result)
            except Exception as e:
                raise _assertion_error(
                    trace,
                    f"tool '{tool_name}' result validation failed (call_id: {call_id}): {e}",
                    expected_tool=tool_name,
                ) from None
        return

    for call_id, result in matched_results:
        context = f"tool '{tool_name}' result field"
        for field, field_schema in schema.items():
            if field not in result:
                raise _assertion_error(
                    trace,
                    f"tool '{tool_name}' result missing field '{field}' (call_id: {call_id})",
                    expected_tool=tool_name,
                )
            _validate_schema(result[field], field_schema, field, trace, context)


def assert_context_preserved(
    source: dict[str, Any], child_trace: Trace, *, fields: list[str]
) -> None:
    """Verify that specific values survived a handoff.

    Compares source[field] == child_trace.handoff_context[field] for each field.
    Fails if field is missing on either side or values differ.
    """
    ctx = _require_handoff_context(child_trace)
    for field in fields:
        if field not in source:
            raise _assertion_error(
                child_trace,
                f"field '{field}' not found in source dict",
            )
        if field not in ctx:
            raise _assertion_error(
                child_trace,
                f"field '{field}' not found in handoff_context",
            )
        src_val = source[field]
        recv_val = ctx[field]
        if src_val != recv_val:
            raise _assertion_error(
                child_trace,
                f"field '{field}' not preserved: source={src_val!r}, received={recv_val!r}",
            )


def assert_called_with(trace: Trace, tool_name: str, **expected_args: Any) -> None:
    """Assert that a tool was called with specific argument values.

    Checks if ``expected_args`` is a subset of any matching tool call's
    arguments. Values are compared with ``==``.
    """
    candidates: list[dict[str, Any]] = []
    best_match_count = -1
    best_mismatch_keys: list[str] = []

    for turn in trace.turns:
        for tc in turn.llm_call.tool_calls:
            if tc.name != tool_name:
                continue
            candidates.append(tc.arguments)
            matched = 0
            mismatched: list[str] = []
            for key, val in expected_args.items():
                if key in tc.arguments and tc.arguments[key] == val:
                    matched += 1
                else:
                    mismatched.append(key)
            if not mismatched:
                return  # full match found
            if matched > best_match_count:
                best_match_count = matched
                best_mismatch_keys = mismatched

    if not candidates:
        raise _assertion_error(
            trace,
            f'"{tool_name}" was never called',
            expected_tool=tool_name,
        )

    raise _assertion_error(
        trace,
        f'"{tool_name}" was called {len(candidates)} time(s) but none matched '
        f"expected args. Closest mismatch on key(s): {best_mismatch_keys}. "
        f"Expected: {expected_args}",
        expected_tool=tool_name,
    )


def _extract_tokens(usage: dict[str, Any]) -> tuple[int, int]:
    """Extract input and output token counts from a token_usage dict.

    Handles both OpenAI (prompt_tokens/completion_tokens) and
    Anthropic (input_tokens/output_tokens) key names. Uses presence-based
    lookup to correctly handle zero values.
    """
    if "input_tokens" in usage:
        input_t = usage["input_tokens"]
    elif "prompt_tokens" in usage:
        input_t = usage["prompt_tokens"]
    else:
        input_t = 0

    if "output_tokens" in usage:
        output_t = usage["output_tokens"]
    elif "completion_tokens" in usage:
        output_t = usage["completion_tokens"]
    else:
        output_t = 0

    return (int(input_t), int(output_t))


def assert_total_tokens_under(trace: Trace, n: int, *, allow_missing: bool = False) -> None:
    """Assert that total token usage across all turns is under n."""
    total = 0
    measured_turns = 0

    for turn in trace.turns:
        usage = turn.llm_call.token_usage
        if usage is None:
            warnings.warn(
                ReagentWarning(
                    f"Turn {turn.index} has no token_usage data — skipped in token count"
                ),
                stacklevel=2,
            )
            continue
        measured_turns += 1
        input_t, output_t = _extract_tokens(usage)
        total += input_t + output_t

    if measured_turns == 0 and trace.turns and not allow_missing:
        raise _assertion_error(
            trace,
            f"Cannot verify token limit: no token data recorded in "
            f"{len(trace.turns)} turn(s). Fix instrumentation or pass "
            f"allow_missing=True.",
        )

    if total > n:
        raise _assertion_error(
            trace,
            f"Total tokens {total} exceeds limit of {n}",
        )


def _match_model_cost(
    model: str, model_costs: dict[str, dict[str, float]]
) -> dict[str, float] | None:
    """Match a model name to costs using longest prefix match."""
    best_prefix = ""
    best_cost: dict[str, float] | None = None
    for prefix, cost in model_costs.items():
        if model.startswith(prefix) and len(prefix) > len(best_prefix):
            best_prefix = prefix
            best_cost = cost
    return best_cost


def assert_cost_under(
    trace: Trace,
    *,
    usd: float,
    model_costs: dict[str, dict[str, float]],
    allow_unpriced: bool = False,
) -> None:
    """Assert that estimated cost is under a USD limit.

    ``model_costs`` maps model name prefixes to per-1M-token prices::

        {"gpt-4o": {"input": 2.50, "output": 10.00}}

    Model names are matched by longest prefix.
    """
    total_cost = 0.0
    priced_turns = 0

    for turn in trace.turns:
        usage = turn.llm_call.token_usage
        model = turn.llm_call.model

        if model is None:
            warnings.warn(
                ReagentWarning(f"Turn {turn.index} has no model set — skipped in cost calculation"),
                stacklevel=2,
            )
            continue

        if usage is None:
            warnings.warn(
                ReagentWarning(
                    f"Turn {turn.index} has no token_usage data — skipped in cost calculation"
                ),
                stacklevel=2,
            )
            continue

        cost_entry = _match_model_cost(model, model_costs)
        if cost_entry is None:
            warnings.warn(
                ReagentWarning(
                    f'Turn {turn.index} model "{model}" not matched by any '
                    f"prefix in model_costs — skipped in cost calculation"
                ),
                stacklevel=2,
            )
            continue

        priced_turns += 1
        input_t, output_t = _extract_tokens(usage)
        total_cost += (input_t / 1_000_000) * cost_entry["input"]
        total_cost += (output_t / 1_000_000) * cost_entry["output"]

    if priced_turns == 0 and trace.turns and not allow_unpriced:
        raise _assertion_error(
            trace,
            f"Cannot verify cost limit: no priced turns in "
            f"{len(trace.turns)} turn(s). Check model_costs keys match your "
            f"models, or pass allow_unpriced=True.",
        )

    if total_cost > usd:
        raise _assertion_error(
            trace,
            f"Estimated cost ${total_cost:.2f} exceeds limit of ${usd:.2f}",
        )
