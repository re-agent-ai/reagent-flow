# reagent-flow-anthropic

Auto-capture Anthropic Python SDK calls into a `reagent-flow` session.

## Install

```bash
pip install reagent-flow-anthropic
```

or

```bash
uv add reagent-flow-anthropic
```

## Usage

```python
from anthropic import Anthropic
from reagent_flow_anthropic import patch
import reagent_flow

client = patch(Anthropic())

with reagent_flow.session("chat") as s:
    client.messages.create(
        model="claude-sonnet-4-0",
        max_tokens=256,
        messages=[{"role": "user", "content": "Look up order 123"}],
        tools=[...],
    )

s.assert_called("lookup_order")
```

`patch()` wraps `client.messages.create` and records tool-use blocks plus
tool results threaded back on later calls.

Full docs: https://github.com/re-agent-ai/reagent-flow
