# reagent-flow

**Contract testing for multi-agent handoffs.** pytest-native assertions that catch schema drift, broken handoffs, and tool-output regressions *before* they ship.

## Why reagent-flow?

Multi-agent systems fail at the seams. One agent hands structured data to the next, and the second agent keeps running even when the shape is subtly wrong — a renamed field, a missing key, a string where an `int` was expected. The LLM papers over it, the tests pass, and the bug surfaces in production as an incoherent decision nobody can trace.

**reagent-flow** treats every handoff as a contract and every tool call as a typed boundary. You declare the schema you expect; reagent-flow records what actually flowed and fails the test when they diverge — with an **Agent Stack Trace** pinpointing the exact field that drifted.

```python
# Assessor receives a handoff from the gatherer. This is the contract:
assessor_session.assert_handoff_matches(schema={
    "release_version": str,
    "ci": {"pipeline": str, "coverage": float},
    "issues": {"open_p0": int, "open_p1": int},
})
```

If the upstream agent renames `open_p0` → `p0_count`, this assertion fails at PR time, not in prod.

### What you get

- **Handoff contracts** — type-check the data passed between agents, with nested dicts, typed lists, and optional Pydantic support.
- **Tool output contracts** — validate the shape of every tool's return value, catching upstream API drift before the downstream agent sees it.
- **Context preservation checks** — verify specific values (IDs, versions, user refs) survive multi-hop handoffs unchanged.
- **Flow and count assertions** — guarantee the tool-calling sequence you expect (order, repetition, forbidden calls).
- **Golden baseline diffs** — snapshot-test known-good traces and detect behavioral regressions from prompt tweaks.
- **Agent Stack Traces** — every failed assertion attaches a readable dump of the full tool-calling history.
- **Zero-dependency core** with thin adapters for OpenAI, Anthropic, LangChain, LangGraph, and CrewAI.

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Session** | A context manager that records tool calls for one agent run; may declare a `parent_trace_id` and `handoff_context` to form a link in a multi-agent chain |
| **Handoff context** | The structured payload passed from one agent session to the next — the target of contract assertions |
| **Trace** | The full sequence of turns captured during a session |
| **Contract** | A declared schema (`{field: type}`) validated against a handoff or tool result |
| **Golden baseline** | A saved trace used as the expected behavior for future runs |
| **Agent Stack Trace** | A readable dump of every turn, attached to assertion failures |

## Installation

```bash
uv add reagent-flow
```

### Framework Adapters

Install only the adapters you need:

```bash
uv add reagent-flow-openai      # OpenAI
uv add reagent-flow-anthropic   # Anthropic
uv add reagent-flow-langchain   # LangChain
uv add reagent-flow-langgraph   # LangGraph
uv add reagent-flow-crewai      # CrewAI
```

## Quick Start

Contract-test a two-agent handoff: the **gatherer** pulls release data, the **assessor** consumes it. reagent-flow validates the shape of what flows between them.

```python
import reagent_flow


def test_release_pipeline(tmp_path):
    trace_dir = str(tmp_path)

    # Phase 1 — gatherer agent records release info
    with reagent_flow.session("gatherer", trace_dir=trace_dir) as gather:
        gather.log_llm_call(
            tool_calls=[{"name": "get_release_info", "arguments": {"version": "v2.3.1"}}],
        )
        gather.log_tool_result(
            "get_release_info",
            result={
                "release_version": "v2.3.1",
                "ci": {"pipeline": "passed", "coverage": 87.3},
                "issues": {"open_p0": 1, "open_p1": 3},
            },
        )

    # Contract on the tool output
    gather.assert_tool_output_matches("get_release_info", schema={
        "release_version": str,
        "ci": {"pipeline": str, "coverage": float},
        "issues": {"open_p0": int, "open_p1": int},
    })

    # Phase 2 — assessor receives the gatherer's output as a handoff
    release_info = gather.trace.turns[0].tool_results[0].result
    with reagent_flow.session(
        "assessor",
        trace_dir=trace_dir,
        parent_trace_id=gather.trace.trace_id,
        handoff_context=release_info,
    ) as assess:
        assess.log_llm_call(
            tool_calls=[{"name": "assess_risk", "arguments": {"risk_level": "HIGH"}}],
        )
        assess.log_tool_result("assess_risk", result={"risk_level": "HIGH"})

    # Contract on the handoff itself — this is where multi-agent systems break
    assess.assert_handoff_received(gather)
    assess.assert_handoff_matches(schema={
        "release_version": str,
        "ci": {"pipeline": str, "coverage": float},
        "issues": {"open_p0": int, "open_p1": int},
    })
    assess.assert_context_preserved({"release_version": "v2.3.1"}, fields=["release_version"])
```

If an upstream change renames `open_p0` → `p0_count`, `assert_handoff_matches` fails with the exact path (`handoff field 'issues.open_p0': missing from data`) attached to an Agent Stack Trace. See the full runnable demo in the repository's `examples/langgraph_demo/` directory.

### Async Support

Sessions work as both sync and async context managers:

```python
async def test_async_agent():
    async with reagent_flow.session("async_flow", trace_dir=".reagent") as s:
        s.log_llm_call(tool_calls=[{"name": "search", "arguments": {"q": "test"}}])
        s.log_tool_result("search", result={"found": True})
    s.assert_called("search")
```

## Assertions

| Method | Description |
|--------|-------------|
| `assert_called(tool)` | Tool was called at least once |
| `assert_never_called(tool)` | Tool was never called |
| `assert_called_before(a, b)` | Tool `a` was called before tool `b` (positional) |
| `assert_tool_succeeded(tool)` | Tool was called and all executions succeeded |
| `assert_max_turns(n)` | Trace has at most `n` turns |
| `assert_total_duration_under(ms=N)` | Total trace duration under `N` ms |
| `assert_matches_baseline()` | Trace matches its golden baseline |
| `assert_flow(pattern)` | Tool calls match a flow pattern (see below) |
| `assert_called_times(tool, min=, max=)` | Tool was called between `min` and `max` times |
| `assert_called_with(tool, **args)` | Tool was called with specific argument values |
| `assert_handoff_received(parent)` | Session is linked to a parent session |
| `assert_handoff_has_fields(fields)` | Required fields exist in handoff context |
| `assert_total_tokens_under(n)` | Total token usage across all turns is under `n` |
| `assert_cost_under(usd=, model_costs=)` | Estimated cost is under a USD limit |
| `assert_handoff_matches(schema=)` | Handoff context matches a `{field: type}` schema |
| `assert_no_extra_fields(allowed=)` | Handoff context has no unexpected fields |
| `assert_tool_output_matches(tool, schema=)` | Tool results match a `{field: type}` schema |
| `assert_context_preserved(source, fields=)` | Specific values survived a handoff unchanged |

## Golden Baselines

Record a golden trace, then assert future runs match:

```python
# Record golden baseline
with reagent_flow.session("flow", golden=True, trace_dir=".reagent") as s:
    run_agent(s)

# Later — assert actual matches golden
with reagent_flow.session("flow", trace_dir=".reagent") as s:
    run_agent(s)
s.assert_matches_baseline()
```

Golden baselines are stored as JSON at `{trace_dir}/golden/{name}.trace.json`. When `assert_matches_baseline()` fails, the diff output shows exactly which tools, arguments, or results changed.

Use `ignore_fields` to skip noisy fields that change between runs:

```python
s.assert_matches_baseline(ignore_fields={"results", "response_text"})
s.assert_matches_baseline(ignore_fields={"lookup.timestamp", "arguments"})
```

Supported values: `"arguments"` (all args), `"results"` (all results), `"response_text"`, or specific keys like `"tool_name.arg_key"`.

## Flow Patterns

`assert_flow` matches tool calls against a pattern with optional gaps using `...` (Ellipsis). Patterns are anchored to start and end by default:

```python
# Exact consecutive match
s.assert_flow(["search", "summarize"])

# Allow any calls between search and summarize
s.assert_flow(["search", ..., "summarize"])

# Unanchored — match anywhere in the trace
s.assert_flow([..., "search", ..., "summarize", ...])
```

## Handoff Integrity

Track parent-child relationships between agent sessions:

```python
with reagent_flow.session("orchestrator") as parent:
    parent.log_llm_call(tool_calls=[{"name": "plan", "arguments": {}}])
    parent.log_tool_result("plan", result="ok")

with reagent_flow.session(
    "researcher",
    parent_trace_id=parent.trace.trace_id,
    handoff_context={"query": "Q3 earnings"},
) as child:
    child.log_llm_call(tool_calls=[{"name": "search", "arguments": {}}])
    child.log_tool_result("search", result="ok")

child.assert_handoff_received(parent)
child.assert_handoff_has_fields(["query"])
```

## Contract Testing

Validate the structure and types of data flowing between agents:

```python
# Type-check handoff context fields
child.assert_handoff_matches(schema={"user_id": str, "query": str, "limit": int})

# Detect unexpected fields leaking through handoffs
child.assert_no_extra_fields(allowed=["user_id", "query", "limit"])

# Validate tool output structure
s.assert_tool_output_matches("search", schema={"results": list, "count": int})

# Verify values survived a handoff unchanged
source = {"user_id": "abc123", "query": "revenue Q4"}
child.assert_context_preserved(source, fields=["user_id", "query"])
```

**Strict bool/int separation:** A field declared as `int` rejects `True`/`False`, and `bool` rejects `0`/`1`. Python's `bool` subclasses `int`, but contract assertions distinguish them.

### Nested Schemas

`assert_handoff_matches` and `assert_tool_output_matches` support nested structures:

```python
# Nested dict
s.assert_handoff_matches(schema={
    "user": {"id": str, "name": str},
    "query": str,
})

# Typed list
s.assert_handoff_matches(schema={"tags": [str]})

# Union typed list
s.assert_handoff_matches(schema={"values": [str, int]})

# List of dicts
s.assert_tool_output_matches("search", schema={
    "results": [{"id": str, "score": float, "title": str}],
    "total": int,
})
```

Error messages include dot/bracket path notation:
- `"handoff field 'user.name': expected str, got int"`
- `"handoff field 'tags[2]': expected str, got int"`
- `"handoff field 'results[0].score': expected float, got str"`

#### Optional Pydantic Support

When Pydantic is installed, pass a `BaseModel` subclass instead of a dict schema:

```python
from pydantic import BaseModel

class HandoffSchema(BaseModel):
    user_id: str
    query: str
    tags: list[str]

child.assert_handoff_matches(schema=HandoffSchema)
```

Pydantic is never imported at module level — detection is purely runtime. Users without Pydantic get identical behavior using dict schemas.

## Token and Cost Guards

Guard against runaway token usage or unexpected costs:

```python
# Assert total tokens across all turns
s.assert_total_tokens_under(50_000)

# Assert estimated cost with per-model pricing (USD per 1M tokens)
s.assert_cost_under(
    usd=1.00,
    model_costs={
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    },
)
```

Model names are matched by longest prefix, so `"gpt-4o"` matches `"gpt-4o-2024-08-06"`.

## Agent Stack Traces

When an assertion fails, reagent-flow attaches a readable stack trace showing every turn, tool call, and result:

```
AGENT STACK TRACE — refund_flow
================================
Turn 0: lookup_order(order_id="123")
  → {"status": "active", "amount": 49.99}
Turn 1: process_refund(order_id="123", amount=49.99)
  → {"success": true}
Turn 2: [text response] "Refund processed."
================================
✗ "delete_account" was never called (3 turns, 2 tool calls)
```

## pytest Integration

reagent-flow ships as a pytest plugin (auto-loaded via entry point). It provides CLI flags, fixtures, and a marker.

### CLI Flags

```bash
pytest --reagent-record       # Set metadata flag for live recording
pytest --reagent-update       # Re-record golden baselines
pytest --reagent-dir=.reagent  # Override trace directory (default: .reagent)
```

### Fixtures

| Fixture | Type | Description |
|---------|------|-------------|
| `reagent_session` | `Session` | A managed session that reads all CLI flags, applies the `@pytest.mark.reagent` marker, and auto-saves on exit |
| `reagent_dir` | `str` | The `--reagent-dir` value |
| `reagent_record` | `bool` | Whether `--reagent-record` was passed |
| `reagent_update` | `bool` | Whether `--reagent-update` was passed |

### Examples

Using the `reagent_session` fixture (recommended):

```python
def test_refund_flow(reagent_session):
    reagent_session.log_llm_call(
        tool_calls=[{"name": "lookup_order", "arguments": {"id": "123"}}],
    )
    reagent_session.log_tool_result("lookup_order", result={"status": "active"})
    reagent_session.assert_called("lookup_order")
```

Recording a golden baseline:

```python
@pytest.mark.reagent(golden=True)
def test_refund_golden(reagent_session):
    run_agent(reagent_session)
```

Or update all goldens at once with `pytest --reagent-update`.

Manual session management still works:

```python
def test_refund_manual(tmp_path):
    with reagent_flow.session("refund", trace_dir=str(tmp_path)) as s:
        run_agent(s)
    s.assert_matches_baseline()
```

## Framework Adapters

### OpenAI

```python
from openai import OpenAI
from reagent_flow_openai import patch
import reagent_flow

client = patch(OpenAI())

with reagent_flow.session("chat") as s:
    client.chat.completions.create(model="gpt-4o", messages=[...], tools=[...])

s.assert_called("my_tool")
```

The `patch()` function wraps `chat.completions.create` to log every LLM turn into the active session. Tool results you send back on the next `create()` call (as `{"role": "tool", "tool_call_id": ..., "content": ...}` messages) are automatically attached to the turn that requested them, so `assert_tool_output_matches` and the other tool-output contracts work end-to-end. JSON-encoded tool content is decoded before validation.

### Anthropic

```python
from anthropic import Anthropic
from reagent_flow_anthropic import patch
import reagent_flow

client = patch(Anthropic())

with reagent_flow.session("chat") as s:
    client.messages.create(model="claude-sonnet-4-20250514", messages=[...], tools=[...], max_tokens=1024)

s.assert_called("my_tool")
```

The `patch()` function wraps `messages.create` to log `tool_use` content blocks on each turn. Tool results you thread back on the next `messages.create` call (as user messages whose `content` is a list of `{"type": "tool_result", "tool_use_id": ..., "content": ...}` blocks) are automatically attached to the turn that requested them, so `assert_tool_output_matches` works end-to-end. JSON-encoded tool content is decoded before validation.

### LangChain

```python
from reagent_flow_langchain import ReagentCallbackHandler
import reagent_flow

handler = ReagentCallbackHandler()

with reagent_flow.session("chain") as s:
    chain.invoke({"input": "..."}, config={"callbacks": [handler]})

s.assert_called("my_tool")
```

### LangGraph

```python
from reagent_flow_langgraph import ReagentGraphTracer
import reagent_flow

tracer = ReagentGraphTracer()

with reagent_flow.session("graph") as s:
    graph.invoke({"input": "..."}, config={"callbacks": [tracer]})

s.assert_called("my_tool")
```

Extends the LangChain handler with graph node tracking.

### CrewAI

```python
from reagent_flow_crewai import instrument
import reagent_flow

crew = instrument(my_crew)

with reagent_flow.session("crew") as s:
    crew.kickoff()

s.assert_called("my_tool")
```

The `instrument()` function wraps all agent tools to capture calls and results into the active session.

## Development

Requires [uv](https://docs.astral.sh/uv/) for package management.

```bash
# Clone and setup
git clone https://github.com/reagent-flow/reagent-flow.git
cd reagent-flow
uv sync  # creates venv, installs all packages + dev deps

# Run tests
uv run pytest packages/ -v

# Run tests with coverage
uv run coverage run -m pytest packages/
uv run coverage report

# Lint and format
uv run ruff check packages/ examples/
uv run ruff format --check packages/ examples/

# Type check
uv run mypy packages/reagent-flow/src/reagent_flow/ --strict
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Security & Privacy

Traces are saved as plain JSON files containing the full tool call arguments and results from your agent runs. **This may include sensitive data** such as API keys, user PII, database contents, or any other values your tools handle.

Before committing traces to version control or sharing them:
- Review trace files for sensitive content
- Add `.reagent/` to your `.gitignore` (golden baselines may be an exception if they contain only synthetic data)
- Consider sanitizing tool inputs/outputs before logging if your agent handles real user data

A built-in redaction framework is planned for a future release.

## Requirements

- Python 3.10+
- Zero runtime dependencies (core library)
- Adapters depend only on their respective framework

## License

[MIT](LICENSE)
