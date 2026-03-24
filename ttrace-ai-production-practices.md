# reagent-ai Production Practices

## Code Quality

- `mypy --strict` on all packages — no untyped `Any` escapes without justification
- `ruff` for lint + format (replaces black/isort/flake8)
- Pre-commit hooks: ruff + mypy + tests before every commit
- Docstrings on all public APIs (ruff `D` rules)
- `py.typed` marker (PEP 561) in every package

## Testing

- pytest with `--strict-markers`
- Coverage gate 90%+ via `pytest-cov`
- Tests organized: unit, integration, e2e with separate markers
- No real LLM calls in CI — adapters tested with mocks

## Git & CI

- Conventional commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`, `ci:`)
- Branch protection on `main` — PRs only, no direct push
- CI matrix: Python 3.10, 3.11, 3.12
- CI pipeline order: lint → type check → test → coverage → build
- All packages tested in CI (core + all adapters)

## Packaging & Release

- `CHANGELOG.md` (Keep a Changelog format)
- Semantic versioning
- `pyproject.toml` (PEP 621) — no `setup.py`
- `hatch` for builds
- `LICENSE` (MIT)

## OSS Standards

- `CONTRIBUTING.md` — dev setup, PR process, code standards
- `CODE_OF_CONDUCT.md`
- Issue/PR templates in `.github/`
- GitHub Actions for CI

## Security

- `ruff` security rules (`S` category)
- `.gitignore` for `.env`, credentials, `__pycache__`, `.mypy_cache`, `.ruff_cache`, `dist/`, `*.egg-info`
- Dependabot config for dependency updates
- No secrets in code

## Naming

- Package name: `reagent-ai` (formerly `ttrace-ai`)
- Python import: `reagent_ai`
- Adapters: `reagent-ai-openai`, `reagent-ai-langchain`, `reagent-ai-langgraph`, `reagent-ai-crewai`
- Spec/plan files reference `agenttrace` — always use `reagent-ai` / `reagent_ai` in code
