.PHONY: lint format test clean help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

lint: ## Run linters (ruff)
	ruff check .

format: ## Auto-format code (black + isort)
	black .
	isort .

test: ## Run the test suite
	pytest tests/ -v

clean: ## Remove caches, .DS_Store, and compiled bytecode
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name .DS_Store -delete 2>/dev/null || true
	rm -rf .pytest_cache
