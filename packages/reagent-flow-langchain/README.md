# reagent-flow-langchain

Trace LangChain tool calls and results into a `reagent-flow` session.

## Install

```bash
pip install reagent-flow-langchain
```

or

```bash
uv add reagent-flow-langchain
```

## Usage

```python
import reagent_flow
from reagent_flow_langchain import ReagentCallbackHandler

handler = ReagentCallbackHandler()

with reagent_flow.session("agent") as s:
    agent.invoke({"input": "Check release risk"}, config={"callbacks": [handler]})

s.assert_called("get_release_info")
```

Full docs: https://github.com/re-agent-ai/reagent-flow
