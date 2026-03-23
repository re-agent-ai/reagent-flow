# Contributing to ttrace-ai

## Development Setup

```bash
# Clone the repo
git clone https://github.com/your-org/ttrace-ai.git
cd ttrace-ai

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install core in dev mode
pip install -e "packages/ttrace-ai[dev]"

# Install adapters
pip install -e packages/ttrace-ai-openai
pip install -e packages/ttrace-ai-langchain
pip install -e packages/ttrace-ai-langgraph
pip install -e packages/ttrace-ai-crewai

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

## Running Tests

```bash
# Core tests
pytest packages/ttrace-ai/tests/ -v

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
mypy packages/ttrace-ai/src/ttrace_ai/ --strict
```

## Conventions

- **Commits**: Use [Conventional Commits](https://www.conventionalcommits.org/) — `feat:`, `fix:`, `test:`, `docs:`, `chore:`, `ci:`
- **Types**: `mypy --strict` on all packages, `py.typed` marker (PEP 561)
- **Docstrings**: All public APIs must have docstrings
- **Tests**: No real LLM calls — adapters use mocks
- **Exceptions**: Inherit from `TTraceError`; warnings use `TTraceWarning` / `TTraceAdapterWarning`

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with tests
3. Ensure all checks pass: `ruff check`, `ruff format --check`, `mypy --strict`, `pytest`
4. Coverage must remain at 90%+
5. Open a PR with a clear description of what and why
