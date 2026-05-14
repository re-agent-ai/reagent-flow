# reagent-flow

**_Contract testing for multi-agent handoffs._**

Catch schema drift, broken handoffs, and tool-output regressions in your test suite — not in production. Pytest-native, zero-dependency core, with thin adapters for OpenAI, Anthropic, LangChain, LangGraph, and CrewAI.

**[Documentation](https://reagent-ai.mintlify.app)** ·
**[Quickstart](https://reagent-ai.mintlify.app/quickstart)** ·
**[Vendor Onboarding showcase](https://reagent-ai.mintlify.app/examples/vendor-onboarding)** ·
**[Discussions](https://github.com/re-agent-ai/reagent-flow/discussions)** ·
**[Changelog](CHANGELOG.md)**

---

## What it looks like

**Vendor Onboarding** example — a multi-agent approval workflow where an intake agent extracts a structured vendor packet and hands it to downstream security, finance, and approval agents. Declare the packet shape once as a contract:

```python
VENDOR_PACKET_SCHEMA = {
    "vendor_name": str,
    "data_access": {
        "contains_customer_pii": bool,
        "data_categories": [str],
        "storage_region": str,
        "retention_days": int,
    },
    "compliance": {
        "soc2_available": bool,
        "dpa_required": bool,
        "subprocessors": [str],
    },
}

security.assert_handoff_matches(schema=VENDOR_PACKET_SCHEMA)
```

When the intake agent's tool drifts — say `contains_customer_pii` is renamed to `handles_personal_data` — the contract fails at the very next boundary, before the security review keeps going on incomplete data:

```text
FAILED test_vendor_onboarding_security_review
  AssertionError: handoff field 'data_access.contains_customer_pii': missing from data

  AGENT STACK TRACE — security
  ─────────────────────────────────────────────────────────────
  parent: intake (a1f2…)
  handoff_context = {
      "vendor_name": "ClearVoice AI",
      "data_access": {
          "handles_personal_data": true,   ← drift
          "data_categories": ["call_audio", "transcripts"],
          "storage_region": "us-east-1",
          "retention_days": 365
      },
      ...
  }

  Turn 0  assess_security_risk(vendor_name="ClearVoice AI")
       ↳ {"risk": "blocked: missing PII flag"}
  ─────────────────────────────────────────────────────────────
```

Read the full walkthrough on the docs site:
[**reagent-ai.mintlify.app/examples/vendor-onboarding**](https://reagent-ai.mintlify.app/examples/vendor-onboarding)

---

## Where reagent-flow fits

| Adjacent tool / approach                 | What it does                                   | Where reagent-flow is different                                       |
| :--------------------------------------- | :--------------------------------------------- | :-------------------------------------------------------------------- |
| **Pydantic AI / structured outputs**     | Validates a single LLM call's output shape.    | Validates the data passed _between_ agents, across multiple sessions. |
| **Guardrails / runtime guards**          | Blocks bad output at runtime, in production.   | Catches it in your test suite, before the PR merges.                  |
| **LangSmith / Langfuse / observability** | Records traces for post-hoc inspection.        | Records _and_ asserts — your CI fails on drift.                       |
| **LLM evals**                            | Scores model output quality on a dataset.      | Asserts deterministic structural contracts on every test run.         |
| **pytest-mock for agents**               | Mocks tool calls so tests don't hit live LLMs. | Captures real or mock traces and asserts on their shape.              |

**Use reagent-flow when you have:**

- Multi-agent or multi-step pipelines passing structured data between sessions
- A pytest suite where you want CI to fail on handoff drift before merge
- Tool outputs whose shape your downstream agents silently depend on

**Reach for something else when:**

- You only need to validate a single LLM call's output → use Pydantic directly
- You need to block bad output at runtime in production → use a guardrails library
- You need accuracy or quality scoring on a dataset → use an evals framework

---

## What's in this monorepo

The core library plus five framework adapters, each a separate installable package:

| Package                  | Version | Purpose                                              | Docs                                                                     |
| :----------------------- | :------ | :--------------------------------------------------- | :----------------------------------------------------------------------- |
| `reagent-flow`           | 0.5.0   | Core: sessions, traces, assertions, golden baselines | [Concepts](https://reagent-ai.mintlify.app/concepts/sessions-and-traces) |
| `reagent-flow-openai`    | 0.2.0   | OpenAI Python SDK adapter                            | [OpenAI](https://reagent-ai.mintlify.app/adapters/openai)                |
| `reagent-flow-anthropic` | 0.2.0   | Anthropic Python SDK adapter                         | [Anthropic](https://reagent-ai.mintlify.app/adapters/anthropic)          |
| `reagent-flow-langchain` | 0.2.0   | LangChain callback handler                           | [LangChain](https://reagent-ai.mintlify.app/adapters/langchain)          |
| `reagent-flow-langgraph` | 0.2.0   | LangGraph callback (extends LangChain)               | [LangGraph](https://reagent-ai.mintlify.app/adapters/langgraph)          |
| `reagent-flow-crewai`    | 0.2.0   | CrewAI tool wrapper                                  | [CrewAI](https://reagent-ai.mintlify.app/adapters/crewai)                |

**Runnable examples** under [`examples/`](examples/):

- [`langgraph_demo/`](examples/langgraph_demo/) — three-agent LangGraph pipeline (Gatherer → Assessor → Decider) that runs end-to-end and demonstrates a broken handoff being caught at the assessor boundary.
- [`manual_logging/`](examples/manual_logging/) — minimal refund flow using explicit `log_llm_call` / `log_tool_result`, no framework adapter required.

---

## Install

```bash
uv add reagent-flow                # core, zero runtime deps
uv add reagent-flow-openai         # +OpenAI
uv add reagent-flow-anthropic      # +Anthropic
uv add reagent-flow-langchain      # +LangChain
uv add reagent-flow-langgraph      # +LangGraph
uv add reagent-flow-crewai         # +CrewAI
```

Python 3.10+. Each adapter depends only on its respective framework.

**Next:** write your first contract test in 5 minutes →
[**reagent-ai.mintlify.app/quickstart**](https://reagent-ai.mintlify.app/quickstart)

---

## Status & roadmap

**Current release:** `reagent-flow 0.5.0`, adapters `0.2.0`. &nbsp; **Stability:** alpha.

**Stable today** (full reference on the [docs site](https://reagent-ai.mintlify.app)):

- Handoff contracts, tool-output contracts, context preservation
- Flow, count, and ordering assertions
- Nested schemas — typed lists, list-of-dicts, optional Pydantic `BaseModel` support
- Golden-baseline diffs with `ignore_fields`
- Token and cost guards with per-model pricing
- Agent Stack Traces attached to every failed assertion
- Five framework adapters with automatic tool-result capture
- pytest plugin: fixtures, marker, CLI flags

**Planned next:**

- Built-in trace redaction framework (for traces that may carry PII or secrets)
- Additional adapters as the community requests them

**Versioning:** while on `0.x`, minor versions may include breaking changes. `1.0` will lock the public assertion API. See [`CHANGELOG.md`](CHANGELOG.md) for what shipped when.

---

## Community

Questions, ideas, war stories about multi-agent handoffs going wrong — all welcome.

- **[GitHub Discussions](https://github.com/re-agent-ai/reagent-flow/discussions)** — Q&A, design conversations, show-and-tell
- **[GitHub Issues](https://github.com/re-agent-ai/reagent-flow/issues)** — bug reports and feature requests
- **[`CONTRIBUTING.md`](CONTRIBUTING.md)** — dev setup, conventions, the 90 % coverage gate
- **[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)** — Contributor Covenant 2.0

Looking to contribute? Start with the [`good first issue`](https://github.com/re-agent-ai/reagent-flow/labels/good%20first%20issue) label.

---

## Development

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/re-agent-ai/reagent-flow.git
cd reagent-flow && uv sync
uv run pytest packages/ -v
uv run ruff check packages/ examples/ && uv run ruff format --check packages/ examples/
uv run mypy packages/reagent-flow/src/reagent_flow/ --strict
```

For architecture notes and contribution guidelines see [`ARCHITECTURE.md`](ARCHITECTURE.md) and [`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## Security & privacy

Traces are plain JSON containing the full tool-call arguments and results from your agent runs — **this may include sensitive data**: API keys, user PII, database contents, anything your tools touch. Review before committing and add `.reagent/` to `.gitignore` unless you're confident the contents are synthetic. A built-in redaction framework is on the roadmap.

For vulnerability disclosure see [`SECURITY.md`](SECURITY.md).

---

## License

[MIT](LICENSE) · [github.com/re-agent-ai/reagent-flow](https://github.com/re-agent-ai/reagent-flow)
