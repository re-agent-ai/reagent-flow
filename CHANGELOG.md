# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Anthropic adapter (`reagent-flow-anthropic`) with `patch()` function for `messages.create`
- Async session support (`async with reagent_flow.session(...) as s:`)
- `ignore_fields` parameter for `assert_matches_baseline()` to skip noisy fields
- `pytest-asyncio` added to dev dependencies
- ARCHITECTURE.md, SECURITY.md, and updated AGENTS.md

### Fixed

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
