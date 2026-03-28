.PHONY: install lint format typecheck test coverage build check clean

install:  ## Install all packages + dev deps, set up pre-commit
	uv sync
	uv run pre-commit install

lint:  ## Run ruff linter
	uv run ruff check packages/ examples/

format:  ## Run ruff formatter
	uv run ruff format packages/ examples/

typecheck:  ## Run mypy strict on all packages
	uv run mypy packages/reagent-flow/src/reagent_flow/ --strict
	uv run mypy packages/reagent-flow-openai/src/reagent_flow_openai/ --strict
	uv run mypy packages/reagent-flow-anthropic/src/reagent_flow_anthropic/ --strict
	uv run mypy packages/reagent-flow-langchain/src/reagent_flow_langchain/ --strict
	uv run mypy packages/reagent-flow-langgraph/src/reagent_flow_langgraph/ --strict
	uv run mypy packages/reagent-flow-crewai/src/reagent_flow_crewai/ --strict

test:  ## Run all tests
	uv run pytest packages/ -v

coverage:  ## Run tests with coverage report
	uv run coverage run -m pytest packages/
	uv run coverage report --show-missing

build:  ## Build all packages
	@for pkg in packages/*/; do \
		echo "Building $$pkg..."; \
		uv build --package $$(basename $$pkg); \
	done

check: lint typecheck test  ## Run all checks (lint + typecheck + test)

clean:  ## Remove build artifacts
	rm -rf dist/ build/
	find packages/ -type d -name __pycache__ -exec rm -rf {} +
	find packages/ -type d -name "*.egg-info" -exec rm -rf {} +

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
