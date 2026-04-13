# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — v1.0 contract testing

### Added

- OpenAI adapter now captures tool results from `{"role": "tool"}` messages on each `create()` call, enabling `assert_tool_output_matches` end-to-end
- Anthropic adapter now captures tool results from `{"type": "tool_result"}` content blocks on each `create()` call
- LangChain adapter `on_tool_end` now unwraps `ToolMessage.content` and JSON-parses dict/list strings before storing
- `_log_prior_tool_results()` + `_parse_tool_content()` helpers in both OpenAI and Anthropic adapters
- `_unwrap_tool_output()` helper in LangChain adapter
- Three-sub-agent Release Gatekeeper demo (`examples/langgraph_demo/`) with Gatherer, Assessor, Decider pipeline
- `orchestrator.py` — runs three sessions in sequence with `parent_trace_id`/`handoff_context` wiring
- `get_release_info_drifted` tool variant for simulating upstream schema drift
- `test_demo.py` — mock-based tests exercising the contract story (no live LLM calls)
- `conftest.py` for demo `--import-mode=importlib` compatibility

### Changed

- Product positioning: reagent-flow = **contract testing for multi-agent handoffs**
- Root README and `packages/reagent-flow/README.md` lede rewritten around contract testing and handoff schemas
- AGENTS.md project summary updated to match new positioning
- Both README adapter sections now describe tool-result capture mechanism
- ARCHITECTURE.md updated to reflect contract testing positioning and adapter tool-result capture
- Rejected three-mode (monitor/shadow/configure) proposal — runtime/policy ideas stay on future roadmap

## [0.4.0] — 2026-04-07

### Added

- Nested schema validation for `assert_handoff_matches` and `assert_tool_output_matches`
- Support for typed lists (`[str]`), union lists (`[str, int]`), and list-of-dicts (`[{"id": str}]`)
- Optional Pydantic `BaseModel` schema support (runtime detection, zero new deps)
- Dot/bracket notation error paths for nested validation errors
- `_require_handoff_context()` shared helper for handoff context validation

### Changed

- Deduplicated handoff context None/dict guards across 4 assertion functions via shared helper
- Hoisted dict-check in `assert_tool_output_matches` before Pydantic/dict branch split
- Removed unused `enumerate` in `assert_flow` loop

## [0.3.0] — 2026-04-07

### Added

- `assert_handoff_matches(trace, schema={"field": type})` — flat type validation with strict bool/int separation
- `assert_no_extra_fields(trace, allowed=["field"])` — strict allowlist check
- `assert_tool_output_matches(trace, "tool", schema={"field": type})` — tool result validation
- `assert_context_preserved(source, trace, fields=["field"])` — value preservation check
- `_strict_isinstance()` helper — treats `bool` and `int` as distinct types in contracts
- Session wrappers for all 4 new assertions

## [0.2.0] - 2026-03-28

### Added

- `assert_flow(pattern)` — pattern matching with `...` (Ellipsis) gaps, anchored start/end
- `assert_called_times(tool, min=, max=)` — bounded call count assertion
- `assert_called_with(tool, **args)` — partial argument matching with closest-mismatch reporting
- Handoff integrity: `parent_trace_id` and `handoff_context` fields on Trace model
- `assert_handoff_received(parent)` / `assert_handoff_has_fields(fields)` — parent-child linking assertions
- `assert_total_tokens_under(n)` — token guard with OpenAI/Anthropic key name support
- `assert_cost_under(usd=, model_costs=)` — cost guard with longest-prefix model matching
- Session wrappers for all new assertions
- Anthropic adapter (`reagent-flow-anthropic`) with `patch()` function for `messages.create`
- Async session support (`async with reagent_flow.session(...) as s:`)
- `ignore_fields` parameter for `assert_matches_baseline()` to skip noisy fields
- `pytest-asyncio` added to dev dependencies
- ARCHITECTURE.md, SECURITY.md, and updated AGENTS.md

### Fixed

- `assert_called_before` now uses flat positional ordering (same-turn parallel calls)
- `__exit__` and `__aexit__` deduplicated via shared `_finalize()` method
- `_all_tool_names` deduplicated via `_flatten_tool_names`
- `assert_total_duration_under()` now uses wall clock for active sessions instead of silently passing
- `log_tool_result()` before any `log_llm_call()` now warns instead of silently dropping
- Diff engine compares all tool calls per turn (not just the first)
- Diff engine compares tool results by position (not by random call_id)
- Path traversal prevention via `_sanitize_name()` in storage backend
- Non-serializable values warn via `ReagentWarning` instead of silent `default=str`
- `assert_tool_succeeded()` now checks for matching results per call_id
- OpenAI and Anthropic adapters detect `stream=True` and skip capture with warning
- Pre-commit pytest hook uses installed packages (activates pytest11 entry point)
- Anthropic adapter included in CI and pre-commit checks

### Changed

- Type cleanup: `Any` → concrete types in session.py (`TracebackType`, `contextvars.Token`) and pytest_plugin.py (`pytest.Parser`, `pytest.Config`)

## [0.1.0] - 2026-03-23

### Changed

- Renamed project from `ttrace-ai` to `reagent-flow`
- All imports changed: `ttrace_ai` → `reagent_flow`, adapters follow same pattern
- Trace directory default changed: `.ttrace/` → `.reagent/`
- CLI flags changed: `--ttrace-*` → `--reagent-*`
- Exception classes renamed: `TTraceError` → `ReagentError`, `TTraceAdapterWarning` → `ReagentAdapterWarning`

### Added

- Core library: session context manager, recorder, trace models
- Assertions engine: `assert_called`, `assert_never_called`, `assert_called_before`, `assert_tool_succeeded`, `assert_max_turns`, `assert_total_duration_under`
- Agent Stack Trace formatter with probable cause heuristics
- Golden baseline diff engine for behavioral regression detection
- JSON storage backend with `.trace.json` format
- pytest plugin with `--reagent-record`, `--reagent-update`, `--reagent-dir` flags
- OpenAI adapter (patch-based instrumentation)
- LangChain adapter (callback handler with call_id tracking)
- LangGraph adapter (extends LangChain with node tracking)
- CrewAI adapter (monkey-patch instrumentation)
- Usage examples
- CI pipeline with lint, type check, test, coverage, and build steps
