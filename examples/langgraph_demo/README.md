# Release Risk Gatekeeper — reagent-flow + LangGraph Demo

Three cooperating sub-agents evaluate whether a software release is safe
to deploy. Each phase is its own reagent-flow session, linked via
`parent_trace_id` and `handoff_context`. reagent-flow's contract
assertions validate the shape of data flowing between agents — which is
where multi-agent systems actually break.

```
        get_release_info                assess_risk              make_decision
User -> Gatherer agent --[handoff]--> Assessor agent --[handoff]--> Decider agent
            |                            |                             |
        session A                    session B                     session C
        (parent)                    (child of A)                 (child of B)
```

## Setup

1. Install dependencies from the repo root:

   ```bash
   uv sync --extra demo
   ```

2. Get a free Gemini API key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

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

`test_demo.py` exercises the same contract-assertion API the demo
showcases but builds the three sessions by hand with `log_llm_call` /
`log_tool_result`, so it runs in CI with **no API key and no live LLM
calls** (per the repo's no-real-LLM-in-tests rule). For the live
end-to-end version, run `demo.py`.

```bash
cd examples/langgraph_demo
uv run pytest test_demo.py -v
```

## What You'll See

### Scenario 1 — Green Path

The full pipeline runs for release `v2.3.1` (risky: open P0, recent
rollback). Each phase's contract passes:

- `gatherer.assert_tool_output_matches("get_release_info", schema=...)`
- `assessor.assert_handoff_received(gatherer)`
- `assessor.assert_handoff_matches(schema=release_info_schema)`
- `assessor.assert_called_times("assess_risk", min=1, max=1)`
- `decider.assert_handoff_received(assessor)`
- `decider.assert_handoff_matches(schema=risk_assessment_schema)`
- `decider.assert_context_preserved({"release_version": "v2.3.1"}, fields=[...])`
- `decider.assert_tool_output_matches("make_decision", schema=...)`

The decider blocks the release. Each session's trace is recorded as a
golden baseline.

### Scenario 2 — Broken Handoff Caught Before Release

The gatherer is swapped for a drifted variant that renames
`issues.open_p0` → `issues.p0_count` and drops `issues.open_p1`
(simulating an upstream refactor gone wrong). The downstream assessor
still runs — the LLM is lenient about missing fields — but
`assessor.assert_handoff_matches(...)` rejects the drifted payload with
an Agent Stack Trace naming the exact missing field:

```
handoff field 'issues.open_p0': missing from data
```

In CI this fails the test and the PR can't merge until the contract or
the gatherer is fixed. **This is the core "caught before release"
story.**

### Scenario 3 — Regression Diff

The pipeline runs again against release `v2.4.0` (clean). The assessor
session is diffed against the golden baseline from scenario 1. The diff
output shows what changed between the two runs — expected here, but in
real CI this is how you catch unintended behavioral regressions from a
prompt tweak or model upgrade.

## Files

| File | Purpose |
|------|---------|
| `tools.py` | The three typed tools, plus a deliberately drifted `get_release_info_drifted` |
| `agent.py` | Sub-agent builders: `build_gatherer_agent`, `build_assessor_agent`, `build_decider_agent` |
| `orchestrator.py` | `run_pipeline` — runs the three sessions in sequence and wires handoff contexts |
| `demo.py` | Three scripted scenarios |
| `test_demo.py` | Same scenarios expressed as pytest tests |

## What reagent-flow is checking

| Check | API | What it catches |
|-------|-----|-----------------|
| Tool output shape | `assert_tool_output_matches` | Upstream API/fixture drift before it reaches another agent |
| Handoff schema | `assert_handoff_matches` | Renamed, missing, or wrongly-typed fields at agent boundaries |
| Parent linkage | `assert_handoff_received` | Child session isn't actually linked to its expected parent |
| Value preservation | `assert_context_preserved` | A key field (version, user id, request id) got lost between hops |
| Call counts | `assert_called_times` | Sub-agent skipped its one job or called it more than once |

Together these turn a fuzzy multi-agent system into something you can
pytest.
