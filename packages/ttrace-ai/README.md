# ttrace-ai

**Record. Assert. Diff.** — Behavioral testing for AI agent tool-calling loops, so a prompt tweak never silently breaks your agent again.

## Why ttrace-ai?

AI agents that call tools are hard to test. A single prompt change can alter which tools get called, in what order, and with what arguments. Traditional unit tests can't catch these regressions because the behavior lives in the LLM's tool-calling loop, not in your code.

**ttrace-ai** records every tool call your agent makes, then lets you assert on the sequence — which tools were called, in what order, whether they succeeded, and how long it all took. When an assertion fails, you get an **Agent Stack Trace** showing the full tool-calling history so you can see exactly where behavior diverged.

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
pip install ttrace-ai
```

### Framework Adapters

Install only the adapters you need:

```bash
pip install ttrace-ai-openai      # OpenAI
pip install ttrace-ai-langchain   # LangChain
pip install ttrace-ai-langgraph   # LangGraph
pip install ttrace-ai-crewai      # CrewAI
```

## Quick Start

```python
import ttrace_ai


def test_my_agent(tmp_path):
    with ttrace_ai.session("order_flow", trace_dir=str(tmp_path)) as s:
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
with ttrace_ai.session("flow", golden=True, trace_dir=".ttrace") as s:
    run_agent(s)

# Later — assert actual matches golden
with ttrace_ai.session("flow", trace_dir=".ttrace") as s:
    run_agent(s)
s.assert_matches_baseline()
```

Golden baselines are stored as JSON at `{trace_dir}/golden/{name}.trace.json`. When `assert_matches_baseline()` fails, the diff output shows exactly which tools, arguments, or results changed.

## Agent Stack Traces

When an assertion fails, ttrace-ai attaches a readable stack trace showing every turn, tool call, and result:

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

ttrace-ai ships as a pytest plugin (auto-loaded via entry point). It adds three CLI flags:

```bash
pytest --ttrace-record       # Force live recording (ignore cached traces)
pytest --ttrace-update       # Re-record golden baselines
pytest --ttrace-dir=.ttrace  # Override trace directory (default: .ttrace)
```

Use the `@pytest.mark.ttrace` marker to tag tests:

```python
@pytest.mark.ttrace(golden=True)
def test_refund_flow(tmp_path):
    with ttrace_ai.session("refund", trace_dir=str(tmp_path)) as s:
        run_agent(s)
    s.assert_matches_baseline()
```

## Framework Adapters

### OpenAI

```python
from openai import OpenAI
from ttrace_ai_openai import patch
import ttrace_ai

client = patch(OpenAI())

with ttrace_ai.session("chat") as s:
    client.chat.completions.create(model="gpt-4o", messages=[...], tools=[...])

s.assert_called("my_tool")
```

The `patch()` function wraps `chat.completions.create` to automatically log tool calls and results into the active session.

### LangChain

```python
from ttrace_ai_langchain import TTraceCallbackHandler
import ttrace_ai

handler = TTraceCallbackHandler()

with ttrace_ai.session("chain") as s:
    chain.invoke({"input": "..."}, config={"callbacks": [handler]})

s.assert_called("my_tool")
```

### LangGraph

```python
from ttrace_ai_langgraph import TTraceGraphTracer
import ttrace_ai

tracer = TTraceGraphTracer()

with ttrace_ai.session("graph") as s:
    graph.invoke({"input": "..."}, config={"callbacks": [tracer]})

s.assert_called("my_tool")
```

Extends the LangChain handler with graph node tracking.

### CrewAI

```python
from ttrace_ai_crewai import instrument
import ttrace_ai

crew = instrument(my_crew)

with ttrace_ai.session("crew") as s:
    crew.kickoff()

s.assert_called("my_tool")
```

The `instrument()` function wraps all agent tools to capture calls and results into the active session.

## Development

```bash
# Clone and install all packages in dev mode
git clone https://github.com/ttrace-ai/ttrace-ai.git
cd ttrace-ai
pip install -e "packages/ttrace-ai[dev]"
pip install -e packages/ttrace-ai-openai \
  -e packages/ttrace-ai-langchain \
  -e packages/ttrace-ai-langgraph \
  -e packages/ttrace-ai-crewai

# Run tests
pytest packages/ examples/ -v

# Run tests with coverage
coverage run -m pytest packages/ examples/ -v
coverage report

# Lint and format
ruff check packages/ examples/
ruff format --check packages/ examples/

# Type check
mypy packages/ttrace-ai/src/ttrace_ai/ --strict
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Requirements

- Python 3.10+
- Zero runtime dependencies (core library)
- Adapters depend only on their respective framework

## License

[MIT](LICENSE)
