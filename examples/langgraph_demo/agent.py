"""Shared agent builder and runner for the Release Risk Gatekeeper demo."""

from __future__ import annotations

DEFAULT_SYSTEM_PROMPT = (
    "You are a Release Risk Gatekeeper. Your job is to evaluate whether "
    "a software release is safe to deploy to production.\n\n"
    "For EVERY release review, you MUST follow this exact process:\n"
    "1. Call get_release_info to gather release data\n"
    "2. Call assess_risk with the release data to produce a risk assessment\n"
    "3. Call make_decision with the risk assessment to make a final APPROVE/BLOCK decision\n\n"
    "Never skip a step. Never make a decision without assessing risk first."
)


def build_agent(system_prompt: str | None = None):
    """Build a LangGraph ReAct agent with the 3 gatekeeper tools."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langgraph.prebuilt import create_react_agent
    from tools import assess_risk, get_release_info, make_decision

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    tools = [get_release_info, assess_risk, make_decision]
    return create_react_agent(llm, tools, prompt=system_prompt or DEFAULT_SYSTEM_PROMPT)


def run_agent(agent, task: str):
    """Run the agent with reagent-flow tracing via ReagentGraphTracer callback."""
    from reagent_flow_langgraph import ReagentGraphTracer

    tracer = ReagentGraphTracer()
    return agent.invoke(
        {"messages": [("user", task)]},
        config={"callbacks": [tracer]},
    )
