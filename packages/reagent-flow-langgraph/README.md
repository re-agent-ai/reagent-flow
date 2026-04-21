# reagent-flow-langgraph

Trace LangGraph agent runs into a `reagent-flow` session.

## Install

```bash
pip install reagent-flow-langgraph
```

or

```bash
uv add reagent-flow-langgraph
```

## Usage

```python
import reagent_flow
from reagent_flow_langgraph import ReagentGraphTracer

tracer = ReagentGraphTracer()

with reagent_flow.session("graph") as s:
    graph.invoke(
        {"messages": [("user", "Check release risk")]},
        config={"callbacks": [tracer]},
    )

s.assert_called("get_release_info")
```

Full docs: https://github.com/re-agent-ai/reagent-flow
