# Contributing to reagent-ai

## Development Setup

```bash
# Clone the repo
git clone https://github.com/your-org/reagent-ai.git
cd reagent-ai

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install core in dev mode
pip install -e "packages/reagent-ai[dev]"

# Install adapters
pip install -e packages/reagent-ai-openai
pip install -e packages/reagent-ai-langchain
pip install -e packages/reagent-ai-langgraph
pip install -e packages/reagent-ai-crewai

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

## Running Tests

```bash
# Core tests
pytest packages/reagent-ai/tests/ -v

# All tests
pytest packages/ examples/ -v

# With coverage
coverage run -m pytest packages/ examples/
coverage report --show-missing
```

## Code Quality

```bash
# Lint
ruff check packages/ examples/

# Format
ruff format packages/ examples/

# Type check
mypy packages/reagent-ai/src/reagent_ai/ --strict
```

## Conventions

- **Commits**: Use [Conventional Commits](https://www.conventionalcommits.org/) — `feat:`, `fix:`, `test:`, `docs:`, `chore:`, `ci:`
- **Types**: `mypy --strict` on all packages, `py.typed` marker (PEP 561)
- **Docstrings**: All public APIs must have docstrings
- **Tests**: No real LLM calls — adapters use mocks
- **Exceptions**: Inherit from `ReagentError`; warnings use `ReagentWarning` / `ReagentAdapterWarning`

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with tests
3. Ensure all checks pass: `ruff check`, `ruff format --check`, `mypy --strict`, `pytest`
4. Coverage must remain at 90%+
5. Open a PR with a clear description of what and why
