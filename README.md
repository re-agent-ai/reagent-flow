# reagent-ai

**Record. Assert. Diff.** — Behavioral testing for AI agent tool-calling loops, so a prompt tweak never silently breaks your agent again.

## Why reagent-ai?

AI agents that call tools are hard to test. A single prompt change can alter which tools get called, in what order, and with what arguments. Traditional unit tests can't catch these regressions because the behavior lives in the LLM's tool-calling loop, not in your code.

**reagent-ai** records every tool call your agent makes, then lets you assert on the sequence — which tools were called, in what order, whether they succeeded, and how long it all took. When an assertion fails, you get an **Agent Stack Trace** showing the full tool-calling history so you can see exactly where behavior diverged.

**Golden baselines** let you snapshot a known-good trace and diff future runs against it, catching regressions the way snapshot testing catches UI regressions.

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Session** | A context manager that records tool calls for one agent run |
| **Turn** | One step in the agent loop: an LLM call, a tool call, or a tool result |
| **Trace** | The full sequence of turns captured during a session |
| **Golden baseline** | A saved trace used as the expected behavior for future runs |
| **Agent Stack Trace** | A readable dump of every turn, attached to assertion failures |

## Installation

```bash
pip install reagent-ai
```

### Framework Adapters

Install only the adapters you need:

```bash
pip install reagent-ai-openai      # OpenAI
pip install reagent-ai-anthropic   # Anthropic
pip install reagent-ai-langchain   # LangChain
pip install reagent-ai-langgraph   # LangGraph
pip install reagent-ai-crewai      # CrewAI
```

## Quick Start

```python
import reagent_ai


def test_my_agent(tmp_path):
    with reagent_ai.session("order_flow", trace_dir=str(tmp_path)) as s:
        # Run your agent — or log manually:
        s.log_llm_call(
            tool_calls=[{"name": "lookup_order", "arguments": {"order_id": "123"}}],
        )
        s.log_tool_result("lookup_order", result={"status": "active"})
        s.log_llm_call(response_text="Order is active.", tool_calls=[])

    # Assertions
    s.assert_called("lookup_order")
    s.assert_never_called("delete_account")
    s.assert_max_turns(5)
```

### Async Support

Sessions work as both sync and async context managers:

```python
async def test_async_agent():
    async with reagent_ai.session("async_flow", trace_dir=".reagent") as s:
        s.log_llm_call(tool_calls=[{"name": "search", "arguments": {"q": "test"}}])
        s.log_tool_result("search", result={"found": True})
    s.assert_called("search")
```

## Assertions

| Method | Description |
|--------|-------------|
| `assert_called(tool)` | Tool was called at least once |
| `assert_never_called(tool)` | Tool was never called |
| `assert_called_before(a, b)` | Tool `a` was called before tool `b` |
| `assert_tool_succeeded(tool)` | Tool was called and all executions succeeded |
| `assert_max_turns(n)` | Trace has at most `n` turns |
| `assert_total_duration_under(ms=N)` | Total trace duration under `N` ms |
| `assert_matches_baseline()` | Trace matches its golden baseline |

## Golden Baselines

Record a golden trace, then assert future runs match:

```python
# Record golden baseline
with reagent_ai.session("flow", golden=True, trace_dir=".reagent") as s:
    run_agent(s)

# Later — assert actual matches golden
with reagent_ai.session("flow", trace_dir=".reagent") as s:
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

## Agent Stack Traces

When an assertion fails, reagent-ai attaches a readable stack trace showing every turn, tool call, and result:

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

reagent-ai ships as a pytest plugin (auto-loaded via entry point). It provides CLI flags, fixtures, and a marker.

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
    with reagent_ai.session("refund", trace_dir=str(tmp_path)) as s:
        run_agent(s)
    s.assert_matches_baseline()
```

## Framework Adapters

### OpenAI

```python
from openai import OpenAI
from reagent_ai_openai import patch
import reagent_ai

client = patch(OpenAI())

with reagent_ai.session("chat") as s:
    client.chat.completions.create(model="gpt-4o", messages=[...], tools=[...])

s.assert_called("my_tool")
```

The `patch()` function wraps `chat.completions.create` to automatically log tool calls and results into the active session.

### Anthropic

```python
from anthropic import Anthropic
from reagent_ai_anthropic import patch
import reagent_ai

client = patch(Anthropic())

with reagent_ai.session("chat") as s:
    client.messages.create(model="claude-sonnet-4-20250514", messages=[...], tools=[...], max_tokens=1024)

s.assert_called("my_tool")
```

The `patch()` function wraps `messages.create` to automatically log `tool_use` content blocks into the active session.

### LangChain

```python
from reagent_ai_langchain import ReagentCallbackHandler
import reagent_ai

handler = ReagentCallbackHandler()

with reagent_ai.session("chain") as s:
    chain.invoke({"input": "..."}, config={"callbacks": [handler]})

s.assert_called("my_tool")
```

### LangGraph

```python
from reagent_ai_langgraph import ReagentGraphTracer
import reagent_ai

tracer = ReagentGraphTracer()

with reagent_ai.session("graph") as s:
    graph.invoke({"input": "..."}, config={"callbacks": [tracer]})

s.assert_called("my_tool")
```

Extends the LangChain handler with graph node tracking.

### CrewAI

```python
from reagent_ai_crewai import instrument
import reagent_ai

crew = instrument(my_crew)

with reagent_ai.session("crew") as s:
    crew.kickoff()

s.assert_called("my_tool")
```

The `instrument()` function wraps all agent tools to capture calls and results into the active session.

## Development

```bash
# Clone and install all packages in dev mode
git clone https://github.com/reagent-ai/reagent-ai.git
cd reagent-ai
pip install -e "packages/reagent-ai[dev]"
pip install -e packages/reagent-ai-openai \
  -e packages/reagent-ai-anthropic \
  -e packages/reagent-ai-langchain \
  -e packages/reagent-ai-langgraph \
  -e packages/reagent-ai-crewai

# Run tests
pytest packages/ examples/ -v

# Run tests with coverage
coverage run -m pytest packages/ examples/ -v
coverage report

# Lint and format
ruff check packages/ examples/
ruff format --check packages/ examples/

# Type check
mypy packages/reagent-ai/src/reagent_ai/ --strict
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
