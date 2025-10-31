.PHONY: help install install-dev clean test test-unit test-integration test-voice lint format check validate run-demo run-voice-demo structure-test docs

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)LangGraph Voice-Enabled Agent Framework - Development Commands$(NC)"
	@echo ""
	@echo "$(GREEN)Available targets:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

install: ## Install package and dependencies
	@echo "$(BLUE)Installing package...$(NC)"
	pip install -e .
	@echo "$(GREEN)✓ Installation complete$(NC)"

install-dev: ## Install package with development dependencies
	@echo "$(BLUE)Installing package with dev dependencies...$(NC)"
	pip install -e ".[dev]"
	@echo "$(GREEN)✓ Development installation complete$(NC)"

install-all: ## Install package with all optional dependencies
	@echo "$(BLUE)Installing package with all dependencies...$(NC)"
	pip install -e ".[all]"
	@echo "$(GREEN)✓ Full installation complete$(NC)"

clean: ## Clean build artifacts and cache files
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

test: ## Run all tests
	@echo "$(BLUE)Running all tests...$(NC)"
	pytest tests/ -v --cov=. --cov-report=term-missing
	@echo "$(GREEN)✓ Tests complete$(NC)"

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	pytest tests/ -v -m "unit" --cov=. --cov-report=term-missing
	@echo "$(GREEN)✓ Unit tests complete$(NC)"

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	pytest tests/ -v -m "integration" --cov=. --cov-report=term-missing
	@echo "$(GREEN)✓ Integration tests complete$(NC)"

test-voice: ## Run voice/LiveKit tests only
	@echo "$(BLUE)Running voice tests...$(NC)"
	pytest tests/ -v -m "voice" --cov=. --cov-report=term-missing
	@echo "$(GREEN)✓ Voice tests complete$(NC)"

test-fast: ## Run tests without coverage (faster)
	@echo "$(BLUE)Running fast tests...$(NC)"
	pytest tests/ -v
	@echo "$(GREEN)✓ Fast tests complete$(NC)"

lint: ## Run linting checks
	@echo "$(BLUE)Running linting checks...$(NC)"
	@echo "$(YELLOW)Checking with flake8...$(NC)"
	flake8 core/ agents/ tools/ voice/ --max-line-length=100 --extend-ignore=E203,W503
	@echo "$(YELLOW)Checking with mypy...$(NC)"
	mypy core/ agents/ tools/ voice/ --ignore-missing-imports
	@echo "$(GREEN)✓ Linting complete$(NC)"

format: ## Format code with black
	@echo "$(BLUE)Formatting code...$(NC)"
	black core/ agents/ tools/ voice/ examples/ tests/
	@echo "$(GREEN)✓ Formatting complete$(NC)"

check: ## Run format check without modifying files
	@echo "$(BLUE)Checking code format...$(NC)"
	black --check core/ agents/ tools/ voice/ examples/ tests/
	@echo "$(GREEN)✓ Format check complete$(NC)"

validate: structure-test lint ## Run all validation checks
	@echo "$(GREEN)✓ All validation checks passed$(NC)"

structure-test: ## Run structural validation test
	@echo "$(BLUE)Running structural validation...$(NC)"
	python test_structure.py
	@echo "$(GREEN)✓ Structural validation complete$(NC)"

run-demo: ## Run basic demo
	@echo "$(BLUE)Running basic demo...$(NC)"
	python examples/basic_demo.py
	@echo "$(GREEN)✓ Demo complete$(NC)"

run-voice-demo: ## Run voice demo
	@echo "$(BLUE)Running voice demo...$(NC)"
	@echo "$(YELLOW)Note: Requires LiveKit configuration$(NC)"
	python examples/voice_demo.py
	@echo "$(GREEN)✓ Voice demo complete$(NC)"

env-setup: ## Create .env file from template
	@if [ -f .env ]; then \
		echo "$(YELLOW)Warning: .env already exists. Backup created as .env.backup$(NC)"; \
		cp .env .env.backup; \
	fi
	cp .env.example .env
	@echo "$(GREEN)✓ .env file created from template$(NC)"
	@echo "$(YELLOW)Please edit .env with your API keys$(NC)"

env-check: ## Check if required environment variables are set
	@echo "$(BLUE)Checking environment variables...$(NC)"
	@python -c "import os; from dotenv import load_dotenv; load_dotenv(); \
		required = ['OPENAI_API_KEY', 'LANGCHAIN_API_KEY']; \
		missing = [v for v in required if not os.getenv(v)]; \
		print('$(GREEN)✓ All required variables set$(NC)' if not missing else '$(RED)✗ Missing: ' + ', '.join(missing) + '$(NC)')"

docs: ## Generate documentation (placeholder)
	@echo "$(BLUE)Generating documentation...$(NC)"
	@echo "$(YELLOW)Documentation generation not yet implemented$(NC)"
	@echo "$(YELLOW)See README.md and examples/ for usage$(NC)"

build: ## Build distribution packages
	@echo "$(BLUE)Building distribution packages...$(NC)"
	python -m build
	@echo "$(GREEN)✓ Build complete - check dist/ directory$(NC)"

publish-test: build ## Publish to TestPyPI
	@echo "$(BLUE)Publishing to TestPyPI...$(NC)"
	python -m twine upload --repository testpypi dist/*
	@echo "$(GREEN)✓ Published to TestPyPI$(NC)"

publish: build ## Publish to PyPI (use with caution!)
	@echo "$(RED)Publishing to PyPI...$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		python -m twine upload dist/*; \
		echo "$(GREEN)✓ Published to PyPI$(NC)"; \
	else \
		echo "$(YELLOW)Cancelled$(NC)"; \
	fi

dev-setup: install-dev env-setup ## Complete development setup
	@echo "$(GREEN)✓ Development environment ready!$(NC)"
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  1. Edit .env with your API keys"
	@echo "  2. Run 'make validate' to check everything works"
	@echo "  3. Run 'make run-demo' to test the framework"

quick-start: install structure-test ## Quick start validation
	@echo "$(GREEN)✓ Quick start complete!$(NC)"
	@echo "$(YELLOW)Try:$(NC)"
	@echo "  make run-demo        - Run basic demo"
	@echo "  make test            - Run tests"
	@echo "  make validate        - Full validation"

info: ## Show project information
	@echo "$(BLUE)LangGraph Voice-Enabled Agent Framework$(NC)"
	@echo ""
	@echo "$(YELLOW)Version:$(NC) 0.1.0"
	@echo "$(YELLOW)Python:$(NC) >=3.9"
	@echo "$(YELLOW)Repository:$(NC) https://github.com/ScientiaCapital/langgraph-voice-agents"
	@echo ""
	@echo "$(GREEN)Core Components:$(NC)"
	@echo "  • 3 specialized agents (Orchestrator, Executor, Checker)"
	@echo "  • 6 MCP tool adapters"
	@echo "  • LiveKit voice integration"
	@echo "  • SQLite + Redis state management"
	@echo ""
	@echo "$(GREEN)Quick Commands:$(NC)"
	@echo "  make install         - Install package"
	@echo "  make run-demo        - Run demo"
	@echo "  make test            - Run tests"
	@echo "  make help            - Show all commands"
