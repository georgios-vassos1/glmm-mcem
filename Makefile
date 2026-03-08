.PHONY: build test install clear-cache

build:
	uv build

install:
	uv sync

test:
	uv run pytest --cov=glmm_mcem --cov-report=term-missing

clear-cache:
	rm -rf .pytest_cache .ruff_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf dist
