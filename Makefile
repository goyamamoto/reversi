# Simple Makefile to run tests and common tasks

SHELL := /bin/sh

TEST_DIR ?= tests
TEST_PATTERN ?= test_*.py

# Prefer venv's python if available
PYTHON := $(shell [ -x venv/bin/python ] && echo venv/bin/python || command -v python3 || command -v python)

.PHONY: help test watch coverage run train clean

help:
	@echo "Available targets:"
	@echo "  make test       - Run unit tests via unittest"
	@echo "  make watch      - Watch files and re-run tests on change"
	@echo "  make coverage   - Run tests with coverage (requires coverage)"
	@echo "  make run        - Launch the game (requires pygame)"
	@echo "  make train      - Run a short training session (writes weights.json)"
	@echo "  make clean      - Remove caches and temporary files"

test:
	$(PYTHON) -m pytest -q

watch:
	$(PYTHON) scripts/watch_tests.py --tests $(TEST_DIR) --pattern '$(TEST_PATTERN)'

coverage:
	@if command -v coverage >/dev/null 2>&1; then \
		coverage run -m pytest && coverage report -m ; \
	else \
		echo "coverage is not installed. Install with: pip install coverage" ; \
		exit 1 ; \
	fi

run:
	$(PYTHON) main.py

train:
	$(PYTHON) train_q.py --episodes 200 --alpha 0.01 --gamma 0.99 --epsilon 0.2 --eps-decay 0.995 --normalize-features --reward margin --alpha-decay 0.999 --l2 1e-4

clean:
	@find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	@find . -name "*.pyc" -delete
	@rm -f .coverage
