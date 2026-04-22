# Releasing reagent-flow safely

This repository is set up for the safest practical publish flow for a solo
maintainer:

- build in GitHub Actions
- publish with PyPI Trusted Publishing
- no long-lived PyPI API token stored in GitHub
- manual dry run to TestPyPI
- production publish only from a GitHub Release

## One-time security setup

### 1. Secure your accounts

- Enable GitHub 2FA. Prefer a passkey or hardware security key.
- Enable PyPI 2FA and generate recovery codes.
- If you use TestPyPI, enable 2FA there too and save recovery codes.

### 2. Create GitHub environments

Create these repository environments:

- `testpypi`
- `pypi`

Recommended settings:

- `pypi`: require manual approval before deployment
- `pypi`: restrict deployment branches to your default branch / release flow
- `testpypi`: optional manual approval, but allowed for dry runs

GitHub environments documentation:
- https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments

### 3. Register Trusted Publishers on PyPI / TestPyPI

For each project:

- `reagent-flow`
- `reagent-flow-openai`
- `reagent-flow-anthropic`
- `reagent-flow-langchain`
- `reagent-flow-langgraph`
- `reagent-flow-crewai`

Register these publishers:

#### TestPyPI

- Repository owner: `re-agent-ai`
- Repository name: `reagent-flow`
- Workflow filename: `.github/workflows/publish-testpypi.yml`
- Environment name: `testpypi`

#### PyPI

- Repository owner: `re-agent-ai`
- Repository name: `reagent-flow`
- Workflow filename: `.github/workflows/publish-pypi.yml`
- Environment name: `pypi`

Official docs:
- https://docs.pypi.org/trusted-publishers/adding-a-publisher/
- https://docs.pypi.org/trusted-publishers/using-a-publisher/
- https://docs.pypi.org/trusted-publishers/security-model/

## Normal release flow

### 1. Prepare the release

- bump package versions
- update `CHANGELOG.md`
- merge to the default branch

### 2. Verify locally

Run:

```bash
uv run ruff check packages/ examples/
uv run ruff format --check packages/ examples/
uv run mypy packages/reagent-flow/src/reagent_flow/ --strict
uv run coverage run -m pytest packages/
uv run pytest examples/langgraph_demo/test_demo.py -v
uv run python examples/langgraph_demo/showcase.py
```

### 3. Dry run on TestPyPI

Use the GitHub Actions workflow:

- `Publish to TestPyPI`

This is manually triggered with `workflow_dispatch`.

### 4. Publish to PyPI

1. Create a Git tag like `v0.5.0`
2. Push the tag
3. Create/publish a GitHub Release for that tag
4. Approve the `pypi` environment deployment when prompted

The `Publish to PyPI` workflow will:

- check out the tagged code
- build all package distributions
- run `twine check`
- publish with Trusted Publishing

## Security notes

- Do not add a `PYPI_TOKEN` GitHub secret.
- Do not publish from pull requests.
- Do not publish from every push to `master`.
- Keep `id-token: write` only on publish jobs.
- Treat workflow file changes as sensitive.
- If a workflow or maintainer account is compromised, rotate credentials and
  remove the Trusted Publisher until you understand the blast radius.
