# ttrace-ai

Behavioral testing library for AI agent tool-calling loops.

Record tool calls, validate sequences, diff against golden baselines, and get Agent Stack Traces when assertions fail.

## Installation

```bash
pip install ttrace-ai
```

### Framework Adapters

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

## Agent Stack Traces

When an assertion fails, ttrace-ai attaches a readable stack trace showing every turn, tool call, and result — plus probable cause detection:

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

## OpenAI Adapter

```python
from openai import OpenAI
from ttrace_ai_openai import patch
import ttrace_ai

client = patch(OpenAI())

with ttrace_ai.session("chat") as s:
    client.chat.completions.create(model="gpt-4o", messages=[...], tools=[...])

s.assert_called("my_tool")
```

## LangChain Adapter

```python
from ttrace_ai_langchain import TTraceCallbackHandler
import ttrace_ai

handler = TTraceCallbackHandler()

with ttrace_ai.session("chain") as s:
    chain.invoke({"input": "..."}, config={"callbacks": [handler]})

s.assert_called("my_tool")
```

## Development

```bash
# Install all packages in dev mode
pip install -e packages/ttrace-ai -e packages/ttrace-ai-openai \
  -e packages/ttrace-ai-langchain -e packages/ttrace-ai-langgraph \
  -e packages/ttrace-ai-crewai

# Run tests
pytest packages/ttrace-ai/tests/ -v

# Lint & type check
ruff check packages/
mypy packages/ttrace-ai/src/ttrace_ai/ --strict
```

## License

MIT
