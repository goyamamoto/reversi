# Simple Makefile to run tests and common tasks

SHELL := /bin/sh

TEST_DIR ?= tests
TEST_PATTERN ?= test_*.py

# Prefer venv's python if available
PYTHON := $(shell [ -x venv/bin/python ] && echo venv/bin/python || command -v python3 || command -v python)

.PHONY: help test test-all watch coverage run train eval auto-tune clean

help:
	@echo "Available targets:"
	@echo "  make test       - Run unit tests via unittest"
	@echo "  make test-all   - Run all tests including integration"
	@echo "  make watch      - Watch files and re-run tests on change"
	@echo "  make coverage   - Run tests with coverage (requires coverage)"
	@echo "  make run        - Launch the game (requires pygame)"
	@echo "  make train      - Run a short training session (writes weights.json)"
	@echo "  make train-auto - Continue from latest, save versioned weights"
	@echo "  make eval       - Evaluate latest weights vs pool/heuristic"
	@echo "  make auto-tune  - Auto tune training/eval cycles (can stop when corner preference emerges)"
	@echo "  make clean      - Remove caches and temporary files"

# By default, exclude integration tests to keep 'make test' fast
PYTEST_MARK ?= -m "not integration"

test:
	$(PYTHON) -m pytest $(PYTEST_MARK) $(PYTEST_ADDOPTS)

test-all:
	$(PYTHON) -m pytest $(PYTEST_ADDOPTS)

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

train-auto:
	$(PYTHON) train_q.py --resume-latest --auto-version --save-dir weights --out weights.json \
	  --episodes 1000 --alpha 0.005 --alpha-decay 0.999 --gamma 0.99 --epsilon 0.2 --eps-decay 0.995 \
	  --normalize-features --reward winloss --random-openings 8 --l2 1e-4 --clip-params 1.0 --checkpoint 500

# Eval parameters (overridable):
EVAL_MATCHES ?= 50
EVAL_DEPTH ?= 4
EVAL_RANDOM_OPENINGS ?= 6
EVAL_OPPONENT ?= pool
EVAL_OPP_DIR ?= weights
EVAL_OPP_ORDER ?= newest
EVAL_OPP_LIMIT ?= 0
EVAL_MATCHES_PER_OPP ?= 0
EVAL_WORKERS ?= 1

eval:
	$(PYTHON) eval_agents.py --matches $(EVAL_MATCHES) --depth $(EVAL_DEPTH) --random-openings $(EVAL_RANDOM_OPENINGS) \
	  --opponent $(EVAL_OPPONENT) --opp-dir $(EVAL_OPP_DIR) --opp-order $(EVAL_OPP_ORDER) \
	  --opp-limit $(EVAL_OPP_LIMIT) --matches-per-opp $(EVAL_MATCHES_PER_OPP) --workers $(EVAL_WORKERS) --progress

# Auto-tune parameters (overridable)
AUTO_TUNE_EVAL_WORKERS ?= 1

auto-tune:
	PYTHONUNBUFFERED=1 $(PYTHON) auto_tuner.py --cycles 5 --episodes 1000 --normalize-features --until-corner --continue-after-corner --corner-threshold 0.7 --progress --eval-workers $(AUTO_TUNE_EVAL_WORKERS)

clean:
	@find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	@find . -name "*.pyc" -delete
	@rm -f .coverage
