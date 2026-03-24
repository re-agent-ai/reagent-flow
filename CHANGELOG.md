# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-23

### Changed

- Renamed project from `ttrace-ai` to `reagent-ai` (`pip install reagent-ai`)
- All imports changed: `ttrace_ai` → `reagent_ai`, adapters follow same pattern
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
