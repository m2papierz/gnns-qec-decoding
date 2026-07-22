.PHONY: install setup fmt lint check test clean

PYTHON   ?= python3
UV       ?= uv

# ── Install ──────────────────────────────────────────────────────────────────

install:
	$(UV) sync

# ── Setup (first-time contributor) ───────────────────────────────────────────

setup: install
	$(UV) run pre-commit install
	@echo "  ✓ Pre-commit hooks installed"

# ── Format ───────────────────────────────────────────────────────────────────

fmt:
	$(UV) run ruff check --select I --fix src/ scripts/ tests/
	$(UV) run ruff format src/ scripts/ tests/

# ── Lint ─────────────────────────────────────────────────────────────────────

lint:
	$(UV) run ruff check src/ scripts/ tests/

# ── Type-check ───────────────────────────────────────────────────────────────

check:
	$(UV) run pyright src/ scripts/ tests/

# ── Test ─────────────────────────────────────────────────────────────────────

test:
	$(UV) run pytest

# ── Clean ────────────────────────────────────────────────────────────────────

clean:
	rm -rf dist/ build/ .pytest_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
