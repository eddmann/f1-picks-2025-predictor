.DEFAULT_GOAL := help

.PHONY: help
help: ## Display this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z\/_%-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Setup

.PHONY: install
install: ## Install dependencies
	@uv sync

.PHONY: install/dev
install/dev: ## Install dependencies with dev extras
	@uv sync --all-extras

##@ Development

.PHONY: lint
lint: ## Run ruff linter
	@uv run ruff check src/ tests/

.PHONY: lint/fix
lint/fix: ## Run ruff linter with auto-fix
	@uv run ruff check --fix --unsafe-fixes src/ tests/

.PHONY: format
format: ## Format code with ruff
	@uv run ruff format src/ tests/

.PHONY: format/check
format/check: ## Check code formatting without changes
	@uv run ruff format --check src/ tests/

##@ Testing

.PHONY: test
test: ## Run all tests
	@uv run pytest tests/ -v

.PHONY: test/unit
test/unit: ## Run unit tests only
	@uv run pytest tests/unit/ -v

.PHONY: ci
ci: lint format/check test ## Run all CI checks (lint + format + test)

##@ Data Management

.PHONY: data/status
data/status: ## Show data summary
	@uv run python -m src.data.loaders --summary

.PHONY: data/sessions
data/sessions: ## List available sessions
	@uv run python -m src.data.loaders --list-sessions

.PHONY: data
data: ## Download all historical data (2020-2025)
	@uv run python -m src.data.fastf1_sync --season 2020 --all
	@uv run python -m src.data.fastf1_sync --season 2021 --all
	@uv run python -m src.data.fastf1_sync --season 2022 --all
	@uv run python -m src.data.fastf1_sync --season 2023 --all
	@uv run python -m src.data.fastf1_sync --season 2024 --all
	@uv run python -m src.data.fastf1_sync --season 2025 --up-to-date

# Usage: make data/sync SEASON=2025 ROUND=24
.PHONY: data/sync
data/sync: ## Sync F1 data for specific round (SEASON=2025 ROUND=24)
	@if [ -z "$(SEASON)" ] || [ -z "$(ROUND)" ]; then \
		echo "Usage: make data/sync SEASON=2025 ROUND=24"; \
		exit 1; \
	fi
	@uv run python -m src.data.fastf1_sync --season $(SEASON) --round $(ROUND)

# Usage: make data/sync/2024
.PHONY: data/sync/%
data/sync/%: ## Sync all data for a season (e.g., make data/sync/2024)
	@uv run python -m src.data.fastf1_sync --season $* --all

##@ Model Training

.PHONY: train
train: ## Train all models (optional: RACE=2025-24)
	@uv run python -m src.cli.retrain --type all $(if $(RACE),--race $(RACE),)

.PHONY: train/qualifying
train/qualifying: ## Train qualifying model only (optional: RACE=2025-24)
	@uv run python -m src.cli.retrain --type qualifying $(if $(RACE),--race $(RACE),)

.PHONY: train/race
train/race: ## Train race model only (optional: RACE=2025-24)
	@uv run python -m src.cli.retrain --type race $(if $(RACE),--race $(RACE),)

.PHONY: train/sprint_quali
train/sprint_quali: ## Train sprint qualifying model only (optional: RACE=2025-24)
	@uv run python -m src.cli.retrain --type sprint_quali $(if $(RACE),--race $(RACE),)

.PHONY: train/sprint_race
train/sprint_race: ## Train sprint race model only (optional: RACE=2025-24)
	@uv run python -m src.cli.retrain --type sprint_race $(if $(RACE),--race $(RACE),)

.PHONY: train/status
train/status: ## Show training data status
	@uv run python -m src.cli.retrain --status

# Usage: make train/tune TYPE=qualifying TRIALS=100
.PHONY: train/tune
train/tune: ## Tune hyperparameters with Optuna (TYPE=qualifying|race TRIALS=50)
	@if [ -z "$(TYPE)" ]; then \
		echo "Usage: make train/tune TYPE=qualifying TRIALS=50"; \
		exit 1; \
	fi
	@uv run python -m src.cli.tune_hyperparams --type $(TYPE) --trials $(or $(TRIALS),50)

##@ Model Export

.PHONY: export/onnx
export/onnx: ## Export all models to ONNX format
	@uv run python -m src.cli.export_onnx --type all

# Usage: make export/onnx/qualifying
.PHONY: export/onnx/%
export/onnx/%: ## Export specific model to ONNX (e.g., make export/onnx/qualifying)
	@uv run python -m src.cli.export_onnx --type $*

# Usage: make export/features RACE=2025-24 TYPE=qualifying
.PHONY: export/features
export/features: ## Export features for a race to JSON (RACE=2025-24 TYPE=qualifying)
	@if [ -z "$(RACE)" ] || [ -z "$(TYPE)" ]; then \
		echo "Usage: make export/features RACE=2025-24 TYPE=qualifying"; \
		exit 1; \
	fi
	@uv run python -m src.cli.export_features --race-id $(RACE) --type $(TYPE)

##@ Predictions

# Usage: make predict/qualifying RACE=2025-24
.PHONY: predict/qualifying
predict/qualifying: ## Predict qualifying top-3 (RACE=2025-24)
	@if [ -z "$(RACE)" ]; then \
		echo "Usage: make predict/qualifying RACE=2025-24"; \
		exit 1; \
	fi
	@uv run python -m src.cli.predict --type qualifying --race-id $(RACE)

# Usage: make predict/race RACE=2025-24
.PHONY: predict/race
predict/race: ## Predict race top-3 (RACE=2025-24)
	@if [ -z "$(RACE)" ]; then \
		echo "Usage: make predict/race RACE=2025-24"; \
		exit 1; \
	fi
	@uv run python -m src.cli.predict --type race --race-id $(RACE)

# Usage: make predict/sprint_quali RACE=2025-24
.PHONY: predict/sprint_quali
predict/sprint_quali: ## Predict sprint quali top-3 (RACE=2025-24)
	@if [ -z "$(RACE)" ]; then \
		echo "Usage: make predict/sprint_quali RACE=2025-24"; \
		exit 1; \
	fi
	@uv run python -m src.cli.predict --type sprint_quali --race-id $(RACE)

# Usage: make predict/sprint_race RACE=2025-24
.PHONY: predict/sprint_race
predict/sprint_race: ## Predict sprint race top-3 (RACE=2025-24)
	@if [ -z "$(RACE)" ]; then \
		echo "Usage: make predict/sprint_race RACE=2025-24"; \
		exit 1; \
	fi
	@uv run python -m src.cli.predict --type sprint_race --race-id $(RACE)

# Usage: make predict/explain RACE=2025-24 TYPE=qualifying
.PHONY: predict/explain
predict/explain: ## Predict with SHAP explanation (RACE=2025-24 TYPE=qualifying)
	@if [ -z "$(RACE)" ] || [ -z "$(TYPE)" ]; then \
		echo "Usage: make predict/explain RACE=2025-24 TYPE=qualifying"; \
		exit 1; \
	fi
	@uv run python -m src.cli.predict --type $(TYPE) --race-id $(RACE) --explain

##@ Evaluation

# Usage: make evaluate RACE=2025-24 TYPE=qualifying
.PHONY: evaluate
evaluate: ## Evaluate model on historical race (RACE=2025-24 TYPE=qualifying)
	@if [ -z "$(RACE)" ] || [ -z "$(TYPE)" ]; then \
		echo "Usage: make evaluate RACE=2025-24 TYPE=qualifying"; \
		exit 1; \
	fi
	@uv run python -m src.cli.evaluate --type $(TYPE) --race-id $(RACE)

.PHONY: evaluate/season
evaluate/season: ## Evaluate model on full season (SEASON=2024 TYPE=qualifying)
	@if [ -z "$(SEASON)" ] || [ -z "$(TYPE)" ]; then \
		echo "Usage: make evaluate/season SEASON=2024 TYPE=qualifying"; \
		exit 1; \
	fi
	@uv run python -m src.cli.evaluate --type $(TYPE) --season $(SEASON)

.PHONY: evaluate/baselines
evaluate/baselines: ## Run baseline comparisons (SEASON=2024)
	@uv run python -m src.models.baselines $(if $(SEASON),--season $(SEASON),)

##@ Backtesting

# Usage: make backtest SEASON=2025
.PHONY: backtest
backtest: ## Run season backtest (SEASON=2025 TYPE=all CACHE=0 START=1)
	@if [ -z "$(SEASON)" ]; then \
		echo "Usage: make backtest SEASON=2025 [TYPE=qualifying] [CACHE=1] [START=round]"; \
		exit 1; \
	fi
	@uv run python -m src.cli.backtest --season $(SEASON) \
		$(if $(filter 1,$(CACHE)),--cache,) \
		$(if $(TYPE),--type $(TYPE),) \
		$(if $(START),--start-round $(START),)

.PHONY: backtest/clean
backtest/clean: ## Remove cached backtest models
	@rm -rf models/saved/backtest_cache/
	@echo "Cleaned backtest model cache"

##@ Inference

# Usage: make inference/php FEATURES=2025-24_qualifying.json
.PHONY: inference/php
inference/php: ## Run PHP ONNX inference (FEATURES=<file.json>)
	@if [ -z "$(FEATURES)" ]; then \
		echo "Usage: make inference/php FEATURES=2025-24_qualifying.json"; \
		echo ""; \
		echo "Available feature files:"; \
		ls -1 models/saved/onnx/features/*.json 2>/dev/null | xargs -I {} basename {} || echo "  (none - run: make export/features RACE=2025-24 TYPE=qualifying)"; \
		exit 1; \
	fi
	@docker run --rm -v $(PWD)/models/saved/onnx:/models f1-picks-2025-predictor-php $(FEATURES)

.PHONY: inference/php/build
inference/php/build: ## Build PHP ONNX inference container
	@docker build -t f1-picks-2025-predictor-php inference/php

##@ Cleaning

.PHONY: clean
clean: ## Clean Python cache files
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf htmlcov/ .coverage 2>/dev/null || true
	@echo "Cleaned cache files"

.PHONY: clean/models
clean/models: ## Remove trained models
	@rm -rf models/*.pkl models/*.json models/*.txt 2>/dev/null || true
	@echo "Cleaned models directory"

.PHONY: clean/predictions
clean/predictions: ## Remove saved predictions
	@rm -rf data/predictions/*.json 2>/dev/null || true
	@echo "Cleaned predictions directory"

.PHONY: clean/all
clean/all: clean clean/models clean/predictions ## Clean everything (cache, models, predictions)
