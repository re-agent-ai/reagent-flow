# Architecture

## What is reagent-flow?

reagent-flow is a behavioral testing library for AI agent tool-calling loops. It captures every tool call an agent makes during a run, then lets you assert on the sequence and diff against golden baselines.

Think of it as **snapshot testing, but for AI agent behavior** -- the way Jest snapshots catch UI regressions, reagent-flow catches agent behavior regressions.

## The Problem

AI agents work by calling tools in loops. A customer support bot might call `lookup_order` -> `check_refund_policy` -> `process_refund`. But a single prompt tweak can silently change which tools get called, in what order, with what arguments. Traditional unit tests can't catch this because they test individual functions, not the behavioral sequence of an entire agent loop.

## The Solution: Record. Assert. Diff.

```python
# Record every tool call
with reagent_flow.session("refund_flow") as s:
    agent.run("Process refund for order 123")

# Assert on the sequence
s.assert_called("lookup_order")
s.assert_called_before("lookup_order", "process_refund")
s.assert_tool_succeeded("process_refund")

# Diff against a known-good baseline
s.assert_matches_baseline()
```

When assertions fail, you get an **Agent Stack Trace** -- a readable dump of every turn showing what went wrong and why.

## Project Structure

Monorepo with a zero-dependency core and thin framework adapters:

```
reagent-flow/
  packages/
    reagent-flow/              Core library (zero runtime dependencies)
    reagent-flow-openai/       OpenAI adapter
    reagent-flow-anthropic/    Anthropic adapter
    reagent-flow-langchain/    LangChain adapter
    reagent-flow-langgraph/    LangGraph adapter (extends LangChain)
    reagent-flow-crewai/       CrewAI adapter
  examples/                  Usage examples
```

**Why separate packages?** Users install only what they need. Testing an OpenAI agent shouldn't require installing LangChain.

---

## High-Level Architecture

```
  +-------------------+     +-------------------+     +-----------------+
  |   Your AI Agent   |     |   Your Test Suite  |     |   pytest CLI    |
  |  (OpenAI, Claude, |     |  (pytest, unittest)|     | --reagent-update|
  |   LangChain, etc) |     |                   |     | --reagent-record|
  +--------+----------+     +--------+----------+     +--------+--------+
           |                          |                         |
           v                          v                         v
  +--------+----------+     +---------+---------+     +--------+--------+
  |  Framework Adapter |     |   Session API     |     |  pytest Plugin  |
  |  (thin, ~100 LOC)  |     |  with/async with  |     |  fixtures/flags |
  +--------+----------+     +---------+---------+     +--------+--------+
           |                          |                         |
           |    get_active_session()  |                         |
           +--------->+<--------------+<------------------------+
                      |
               +------+------+
               |  ContextVar  |
               | (thread-safe |
               |  async-safe) |
               +------+------+
                      |
               +------v------+
               |   Session    |
               |  (recorder,  |
               |   trace,     |
               |   assertions)|
               +------+------+
                      |
         +------------+------------+
         |            |            |
  +------v------+ +---v----+ +----v-----+
  |  Recorder   | | Assert | |  Storage  |
  | (builds     | | Engine | | (JSON     |
  |  turns from | |        | |  backend) |
  |  log events)| |        | |           |
  +------+------+ +---+----+ +----+------+
         |             |           |
         v             v           v
  +------+------+ +----+-----+ +--+----------+
  |   Trace     | | Stack    | | File System  |
  | (dataclass  | | Trace    | | traces/      |
  |  hierarchy) | | Formatter| | golden/      |
  +-----------+-+ +----------+ +--+-----------+
              |                    |
              v                    v
        +-----+------+    +-------+------+
        | Diff Engine |    | Golden       |
        | (turn-by-   |    | Baselines    |
        |  turn)      |    | (.trace.json)|
        +-------------+    +--------------+
```

## Core Concepts

### Two Main Flows

```
  Recording Flow                     Assertion Flow
  ==============                     ===============

  Agent calls LLM                    s.assert_called("tool")
       |                                    |
       v                                    v
  Adapter intercepts response        _sync_trace()
       |                             (copy recorder -> trace)
       v                                    |
  get_active_session()                      v
       |                             Pure function on Trace
       v                                    |
  session.log_llm_call()                    v
  session.log_tool_result()          Pass: return None
       |                             Fail: AssertionError
       v                                   + Agent Stack Trace
  Recorder builds Turn                     + Probable Cause
```

### Request Lifecycle (Single Turn)

```
  1. LLM Response Received
     +------------------------------------------------------+
     | response = client.chat.completions.create(...)        |
     +------------------------------------------------------+
                              |
                              v
  2. Adapter Captures
     +------------------------------------------------------+
     | tool_calls = extract_tool_calls(response)             |
     | call_ids = session.log_llm_call(tool_calls=...)       |
     +------------------------------------------------------+
                              |
                              v
  3. Recorder Creates Turn
     +------------------------------------------------------+
     | Turn(index=N)                                         |
     |   LLMCall(tool_calls=[ToolCall(name, args, call_id)]) |
     |   tool_results=[]  <-- pending                        |
     +------------------------------------------------------+
                              |
                              v
  4. Tools Execute & Results Logged
     +------------------------------------------------------+
     | result = execute_tool(name, args)                     |
     | session.log_tool_result(name, call_id=id, result=...) |
     +------------------------------------------------------+
                              |
                              v
  5. Turn Complete
     +------------------------------------------------------+
     | Turn(index=N)                                         |
     |   LLMCall(tool_calls=[ToolCall(...)])                  |
     |   tool_results=[ToolResult(call_id, result, ...)]     |
     +------------------------------------------------------+
```

---

## Data Model

All trace data is plain Python dataclasses with no external dependencies.

```
Trace
  trace_id: str (UUID)
  name: str
  metadata: dict
  started_at: float (epoch)
  ended_at: float | None
  format_version: str
  turns: list[Turn]

Turn
  index: int
  llm_call: LLMCall
  tool_results: list[ToolResult]
  duration_ms: float

LLMCall
  messages: list[Message] | None
  response_text: str | None
  tool_calls: list[ToolCall]
  model: str | None
  token_usage: dict | None
  timestamp: float

ToolCall
  name: str
  arguments: dict
  call_id: str
  timestamp: float

ToolResult
  call_id: str
  result: Any
  error: str | None
  duration_ms: float
```

**Why call_id?** LLMs can request multiple tool calls in a single response (parallel tool calls). When two calls have the same name (e.g., two `lookup` calls), `call_id` is the only way to match each result to the correct call.

---

## Core Components

### Session Lifecycle

```
                    ContextVar
                       |
  with session("name") as s:       # __enter__: sets ContextVar token
      adapter.call(...)            #   adapter reads get_active_session()
      s.log_llm_call(...)          #   manual logging alternative
      s.log_tool_result(...)       #
      s.assert_called("tool")      #   syncs trace, runs assertion
  # __exit__:                      # syncs trace, sets ended_at, resets ContextVar, saves to disk
```

1. **Enter**: `Session.__enter__` sets the session into a `ContextVar` so adapters can find it without being passed a reference.
2. **Record**: `Recorder` builds `Turn` objects from `log_llm_call` / `log_tool_result` events. Each `log_llm_call` creates a new Turn; tool results attach to the current Turn.
3. **Assert**: Assertion methods call `_sync_trace()` to copy the recorder's live turns into the `Trace` dataclass, then run the check. This means assertions work both inside and after the session.
4. **Exit**: Finalizes the trace (sets `ended_at`), resets the `ContextVar`, and saves to disk via the storage backend.

Async sessions (`async with`) follow the same lifecycle.

### Thread Safety

`_context.py` uses `contextvars.ContextVar` for session binding. Each thread or asyncio task gets its own session reference. There is no global mutable state.

```
  Thread/Task 1          Thread/Task 2          Thread/Task 3
  +-------------+        +-------------+        +-------------+
  | ContextVar  |        | ContextVar  |        | ContextVar  |
  | session=A   |        | session=B   |        | session=None|
  +------+------+        +------+------+        +------+------+
         |                      |                      |
         v                      v                      v
  +------+------+        +------+------+        No recording
  | Session A   |        | Session B   |        (adapters skip)
  | Recorder A  |        | Recorder B  |
  +-------------+        +-------------+
```

### Assertion Engine

Assertions are pure functions that operate on a `Trace` dataclass:

| Assertion | Logic |
|-----------|-------|
| `assert_called(tool)` | Scans all turns for a matching tool name |
| `assert_never_called(tool)` | Inverse of `assert_called` |
| `assert_called_before(a, b)` | Compares first occurrence indices |
| `assert_tool_succeeded(tool)` | Checks tool was called AND all results have `error=None` AND matching `call_id` |
| `assert_max_turns(n)` | `len(trace.turns) <= n` |
| `assert_total_duration_under(ms)` | Uses wall clock for active sessions, `ended_at` for finalized |

All assertion failures raise `AssertionError` with an **Agent Stack Trace** attached.

### Agent Stack Trace

When an assertion fails, the full trace is formatted as a readable stack trace with probable cause detection:

```
==================================================
AGENT STACK TRACE -- test_billing_refund
==================================================

+ Turn 0 [120ms]
  LLM -> lookup_order(order_id="123")
  Result: {"status": "shipped"}

+ Turn 1 [85ms]
  LLM -> check_refund_policy(order_id="123")
  Result: {"eligible": false, "reason": "already shipped"}

x Turn 2 [40ms]
  LLM -> [TEXT] "I cannot process a refund."
  No tool calls made.

==================================================
ASSERTION FAILED: assert_called("process_refund")
  "process_refund" was never called (3 turns, 2 tool calls)

PROBABLE CAUSE:
  Turn 1 returned eligible=false -> LLM decided to skip refund
==================================================
```

**Probable cause heuristics** (pure logic, no LLM calls):
- Expected tool never called -> show last tool result before LLM stopped
- Tool errored and LLM stopped -> flag that tool error
- LLM made a text-only response -> show what decision was made

### Golden Baseline Diff

`diff.py` compares two traces turn-by-turn to detect behavioral regressions:

```
  Golden Trace              Actual Trace              DiffResult
  +-------------+          +-------------+          +------------------+
  | Turn 0:     |          | Turn 0:     |          | Turn 0: MATCH    |
  | lookup(123) |  =====>  | lookup(123) |  =====>  |                  |
  +-------------+          +-------------+          +------------------+
  | Turn 1:     |          | Turn 1:     |          | Turn 1: MISMATCH |
  | refund(123) |  =====>  | cancel(123) |  =====>  | expected: refund |
  +-------------+          +-------------+          | actual:   cancel |
  | Turn 2:     |          +-------------+          +------------------+
  | confirm()   |                                   | Turn 2: MISSING  |
  +-------------+                                   +------------------+
```

1. Compares all tool calls per turn by position (not just the first)
2. Compares tool results by position (not by call_id, since call_ids are random UUIDs)
3. Compares response text
4. Supports `ignore_fields` to skip noisy fields: `"arguments"`, `"results"`, `"response_text"`, or specific keys like `"tool_name.arg_key"`

### Storage

Traces are stored as JSON files:

```
{trace_dir}/
  traces/
    {name}_{timestamp}.trace.json     Regular traces (accumulate)
  golden/
    {name}.trace.json                 Golden baselines (one per name, overwritten)
```

**Path traversal prevention**: All trace names are sanitized via `_sanitize_name()` which strips path separators and special characters. Names that sanitize to empty raise `ReagentError`.

**Serialization**: Non-JSON-serializable values are converted to strings with a `ReagentWarning`, rather than silently coercing via `default=str`.

---

## Adapters

All adapters follow the same contract: intercept framework-specific API calls, extract tool call data, and log it into the active session via `get_active_session()`.

```
  +-------------------+        +-------------------+
  |  Original Client   |        |   Patched Client   |
  |                   |  patch  |                   |
  | client.create()   +------->+ client.create()   |
  |   -> response     |        |   -> response     |
  +-------------------+        |   -> log to session|
                               +-------------------+
```

| Adapter | Strategy | What it wraps |
|---------|----------|---------------|
| OpenAI | `patch(client)` | `client.chat.completions.create` |
| Anthropic | `patch(client)` | `client.messages.create` |
| LangChain | Callback handler | `on_llm_end` / `on_tool_end` |
| LangGraph | Extends LangChain | Adds graph node tracking |
| CrewAI | `instrument(crew)` | Monkey-patches all agent tool `_run` methods |

Adapters are separate packages with a single dependency on their framework. They never import each other.

**Error policy**: Adapters never break the original API call. If capture fails, the original response is returned normally and a `ReagentAdapterWarning` is emitted.

**Streaming**: `stream=True` is detected and skipped with a warning. Only synchronous responses are captured.

---

## pytest Plugin

Registered via `[project.entry-points.pytest11]` in pyproject.toml. Provides:

- **CLI flags**: `--reagent-record`, `--reagent-update`, `--reagent-dir`
- **Fixtures**: `reagent_session`, `reagent_dir`, `reagent_record`, `reagent_update`
- **Marker**: `@pytest.mark.reagent(golden=True)`

The `reagent_session` fixture creates a `Session`, applies CLI flags and marker kwargs, and auto-saves on exit.

---

## Package Layout (Detailed)

```
packages/reagent-flow/src/reagent_flow/
  __init__.py           Public API: session(), exceptions, version
  _context.py           ContextVar for thread-safe session binding
  session.py            Session context manager (sync + async)
  recorder.py           Builds Turn objects from log events
  models.py             Dataclasses: Trace, Turn, LLMCall, ToolCall, ToolResult
  assertions.py         Assertion implementations
  stacktrace.py         Agent Stack Trace formatter
  diff.py               Golden baseline diff engine
  exceptions.py         Exception and warning hierarchy
  storage/
    json.py             JSON file storage backend
  pytest_plugin.py      pytest plugin (entry point: pytest11)
  py.typed              PEP 561 type marker
```

---

## Exception Hierarchy

```
Exception
  ReagentError                    Base for all reagent-flow exceptions
    SessionClosedError            log_* on finalized session
    AmbiguousToolCallError        Parallel same-name tool calls without call_id
    TraceNotFoundError            Missing golden baseline (also extends FileNotFoundError)

UserWarning
  ReagentWarning                  Core warnings (serialization, dropped results)
  ReagentAdapterWarning           Adapter capture failures (streaming, etc.)
```

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Zero runtime deps | Avoids conflicts; traces are just dataclasses and JSON |
| ContextVar over global | Thread-safe and asyncio-safe without locks |
| Separate adapter packages | Users install only what they need; no framework dep in core |
| Dataclasses over Pydantic | Lighter, no validation overhead for internal data |
| `_sync_trace()` on every assertion | Enables assertions inside active sessions |
| Positional diff comparison | Tool call order matters for behavioral testing; call_ids are random UUIDs |
| `coverage run` over `--cov` | Avoids plugin load-order issue with pytest entry points |
| Warnings over silent failures | Users can fix what they can see |
| `_sanitize_name()` on all trace names | Prevents path traversal attacks via malicious trace names |
| `format_version` in Trace | Forward compatibility for future schema changes |
