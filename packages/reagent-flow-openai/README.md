# reagent-flow-openai

Auto-capture OpenAI Python SDK calls into a `reagent-flow` session.

## Install

```bash
pip install reagent-flow-openai
```

or

```bash
uv add reagent-flow-openai
```

## Usage

```python
from openai import OpenAI
from reagent_flow_openai import patch
import reagent_flow

client = patch(OpenAI())

with reagent_flow.session("chat") as s:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Look up order 123"}],
        tools=[...],
    )

s.assert_called("lookup_order")
```

`patch()` wraps `client.chat.completions.create` and records tool calls,
response text, token usage, and tool results returned on subsequent
`{"role": "tool"}` messages.

Full docs: https://github.com/re-agent-ai/reagent-flow
