# Release Risk Gatekeeper — reagent-flow + LangGraph Demo

An AI agent evaluates whether a software release is safe to deploy. reagent-flow traces every tool call, asserts on agent behavior, and detects regressions via golden baseline diffs.

## Setup

1. Install dependencies from the repo root:

```bash
uv sync --extra demo
```

2. Get a free Gemini API key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

3. Export the key:

```bash
export GOOGLE_API_KEY="your-key-here"
```

## Run the Demo

```bash
cd examples/langgraph_demo
uv run python demo.py
```

## Run as Tests

```bash
cd examples/langgraph_demo
uv run pytest test_demo.py -v
```

## What You'll See

### Scenario 1: Green Path
The agent evaluates a risky release (failing tests, open P0 bugs, recent rollback) and blocks it. reagent-flow asserts that all 3 tools were called in the correct order.

### Scenario 2: Red Path
The agent handles an emergency hotfix under time pressure. If it skips the risk assessment step, reagent-flow catches it with a failed assertion and Agent Stack Trace. If the agent is diligent, a pre-built example shows what the failure would look like.

### Scenario 3: Diff Path
The agent evaluates a clean release and approves it. reagent-flow diffs this against the golden baseline from Scenario 1, showing how the agent's behavior changed with different input data.

## Tools

| Tool | Purpose |
|------|---------|
| `get_release_info(version)` | Returns realistic CI/CD data (tests, coverage, bugs, deploy history) |
| `assess_risk(release_info)` | LLM reasons about risk level (LOW/MEDIUM/HIGH) |
| `make_decision(risk_assessment)` | LLM decides APPROVE or BLOCK |

## Architecture

```
User -> LangGraph ReAct Agent -> Tools -> reagent-flow traces everything
                                            |
                                    Session records: tool calls, results, LLM responses
                                            |
                                    Assertions verify: ordering, completeness, success
                                            |
                                    Baseline diffs detect: behavioral regressions
```
