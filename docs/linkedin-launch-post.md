LLMs are non-deterministic. Your agent worked yesterday. Today it skipped a step.

Nobody noticed until production broke.

Here's the problem:
- Same prompt, same model, different behavior
- Agent called 3 tools yesterday, 2 today
- Refund got processed without checking the order status
- Risk assessment got skipped before deploy approval

Traditional testing can't catch this. Unit tests check code. But agent behavior isn't code — it's emergent.

So I built reagent-flow.

Think of it as snapshot testing for AI agents.

It records every tool call, every decision, every result. Then it lets you assert:
- "Did the agent check risk before approving?"
- "Did it call all required tools?"
- "Did its behavior change from last week's baseline?"

When something drifts, you get an Agent Stack Trace showing exactly what changed and where.

Here's a real example:

We built a Release Risk Gatekeeper agent. It evaluates whether a deploy is safe.

Run 1: get_release_info → assess_risk → make_decision(BLOCK)
Run 2: get_release_info → make_decision(APPROVE)

The agent skipped risk assessment. In production, nobody would have caught this.

reagent-flow caught it in 3 lines:

assert_called("assess_risk")
assert_called_before("assess_risk", "make_decision")
assert_matches_baseline()

Works with LangChain, LangGraph, OpenAI, Anthropic, CrewAI. Zero dependencies in core. Drop it into any agent with 2 lines of code.

LLMs will always be non-deterministic. Your reliability checks shouldn't be.

reagent-flow is open source → github.com/sssmaran/reagent-flow

---

#AI #LLM #AgentReliability #OpenSource #DevTools #Testing #AIAgents #Python
