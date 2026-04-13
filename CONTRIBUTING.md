# Contributing to reagent-flow

## Development Setup

Requires [uv](https://docs.astral.sh/uv/) for package management.

```bash
# Clone the repo
git clone https://github.com/reagent-flow/reagent-flow.git
cd reagent-flow

# Install everything (creates venv, installs all packages + dev deps)
uv sync

# Install pre-commit hooks
uv run pre-commit install
```

## Running Tests

```bash
# Core tests
uv run pytest packages/reagent-flow/tests/ -v

# All tests
uv run pytest packages/ -v

# With coverage
uv run coverage run -m pytest packages/
uv run coverage report --show-missing
```

## Code Quality

```bash
# Lint
uv run ruff check packages/ examples/

# Format
uv run ruff format packages/ examples/

# Type check
uv run mypy packages/reagent-flow/src/reagent_flow/ --strict
```

## Conventions

- **Commits**: Use [Conventional Commits](https://www.conventionalcommits.org/) -- `feat:`, `fix:`, `test:`, `docs:`, `chore:`, `ci:`
- **Types**: `mypy --strict` on all packages, `py.typed` marker (PEP 561)
- **Docstrings**: All public APIs must have docstrings
- **Tests**: No real LLM calls -- adapters use mocks
- **Exceptions**: Inherit from `ReagentError`; warnings use `ReagentWarning` / `ReagentAdapterWarning`

## Pull Request Process

1. Create a feature branch from `master`
2. Make your changes with tests
3. Ensure all checks pass: `uv run ruff check`, `uv run ruff format --check`, `uv run mypy --strict`, `uv run pytest`
4. Coverage must remain at 90%+
5. Open a PR with a clear description of what and why
