# reagent-flow-crewai

Trace CrewAI tool execution into a `reagent-flow` session.

## Install

```bash
pip install reagent-flow-crewai
```

or

```bash
uv add reagent-flow-crewai
```

## Usage

```python
import reagent_flow
from reagent_flow_crewai import instrument

instrument()

with reagent_flow.session("crew") as s:
    crew.kickoff()

s.assert_called("search_docs")
```

Full docs: https://github.com/re-agent-ai/reagent-flow
