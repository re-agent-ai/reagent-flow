# Vendor Onboarding Showcase

This demo shows an AI vendor-approval workflow with separate intake, security,
finance, and approval agents:

```text
Intake Agent -> Security Review Agent
             -> Finance Review Agent
             -> Approval Agent
```

The scenario is a request to approve ClearVoice AI, a tool that transcribes
customer calls. The intake agent is supposed to hand a structured vendor packet
to security and finance.

## Scenario

The stable contract says security must receive:

- `data_access.contains_customer_pii: bool`
- `compliance.subprocessors: list[str]`

The drifted intake payload changes:

- `contains_customer_pii` -> `handles_personal_data`
- `subprocessors` list -> comma-separated string

Without the handoff contract, the security review treats the vendor as LOW risk
and the approval agent records APPROVE. reagent-flow fails the test before that
approval can ship.

## Run

```bash
cd examples/vendor_onboarding_showcase
uv run python showcase.py
```

Expected punchline:

```text
ASSERTION FAILED: handoff field 'data_access.contains_customer_pii': missing from data
Result: vendor approval blocked before a risky security review reaches the approver.
```

## Test

```bash
cd examples/vendor_onboarding_showcase
uv run pytest test_showcase.py -v
```

The tests prove both sides of the story:

- Clean packet: contracts pass, PII is visible, approval escalates.
- Drifted packet: PII is hidden, the workflow would approve, but reagent-flow
  fails the handoff contract first.
