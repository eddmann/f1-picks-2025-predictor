# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

F1 prediction system that forecasts top-3 finishers across 4 session types using LightGBM LambdaRank models:

- **Qualifying (Q)** - Predict qualifying top-3
- **Sprint Qualifying (SQ)** - Predict sprint qualifying top-3 (sprint weekends only)
- **Sprint Race (S)** - Predict sprint race top-3 (sprint weekends only)
- **Race (R)** - Predict race top-3

Uses a game scoring system: 2 points for exact position match, 1 point for correct driver in wrong position (max 6 points per session).

## Commands

All commands use Make targets. Run `make help` to see all available targets.

```bash
# Setup
make install           # Install dependencies
make install/dev       # Install with dev extras

# Data Management
make data              # Download all historical data (2020-2025)
make data/sync SEASON=2024 ROUND=5   # Sync specific round
make data/sync/2024    # Sync entire season
make data/status       # Show data summary
make data/sessions     # List available sessions

# Training
make train             # Train all 4 model types
make train/qualifying  # Train qualifying model only
make train/race        # Train race model only
make train/sprint_quali
make train/sprint_race
make train/tune TYPE=qualifying TRIALS=50  # Hyperparameter tuning
make train/status          # Show training data status

# Predictions
make predict/qualifying RACE=2025-24
make predict/race RACE=2025-24
make predict/sprint_quali RACE=2025-24  # Sprint weekends only
make predict/sprint_race RACE=2025-24   # Sprint weekends only
make predict/explain RACE=2025-24 TYPE=qualifying  # With SHAP explanation

# Evaluation
make evaluate RACE=2025-24 TYPE=qualifying
make evaluate/season SEASON=2024 TYPE=qualifying
make evaluate/baselines SEASON=2024

# Backtesting
make backtest SEASON=2025              # Run full season backtest
make backtest SEASON=2025 TYPE=qualifying  # Backtest specific model type
make backtest SEASON=2025 START=10     # Start from round 10
make backtest/clean                    # Remove cached backtest models

# Model Export (ONNX)
make export/onnx            # Export all models to ONNX format
make export/onnx/qualifying # Export specific model
make export/features RACE=2025-24 TYPE=qualifying  # Export features to JSON

# PHP Inference
make inference/php FEATURES=2025-24_qualifying.json  # Run PHP prediction
make inference/php/build    # Build PHP Docker image

# Development
make test              # Run all tests
make test/unit         # Run unit tests only
make lint              # Run ruff linter
make lint/fix          # Run linter with auto-fix
make format            # Format code
make format/check      # Check formatting without changes
make ci                # Run all CI checks (lint + format + test)

# Cleaning
make clean             # Clean Python cache files
make clean/models      # Remove trained models
make clean/predictions # Remove saved predictions
make clean/all         # Clean everything
```

## Data Structure

Data is stored as parquet files in `data/fastf1/`:

```
data/fastf1/
├── sessions/           # Per-session lap data
│   ├── 2024_01_FP1.parquet
│   ├── 2024_01_FP2.parquet
│   ├── 2024_01_FP3.parquet
│   ├── 2024_01_Q.parquet
│   └── 2024_01_R.parquet
└── metadata/
    ├── events.parquet  # Race calendar
    ├── drivers.parquet # Driver registry
    └── teams.parquet   # Team registry
```

Session parquet files contain:

- Lap times (total and S1/S2/S3 sectors) in milliseconds
- Tyre compound and life
- Speed trap data
- Qualifying session split (Q1/Q2/Q3)
- Weather conditions
- Final position and points

## Architecture

The system enforces strict temporal integrity: features for Race N only use data from Races 1 to N-1. This prevents data leakage and is validated by tests in `tests/unit/test_fastf1_features.py`.

### Multi-Session Architecture

Each prediction type has its own feature pipeline and model due to different data availability:

**Standard Weekend (FP1 → FP2 → FP3 → Q → R):**

| Predicting | Current Weekend Available     | NOT Available |
| ---------- | ----------------------------- | ------------- |
| **Q**      | FP1, FP2, FP3, historical     | -             |
| **R**      | FP1, FP2, FP3, Q grid, hist.  | -             |

**Sprint Weekend (FP1 → SQ → S → Q → R):**

| Predicting | Current Weekend Available        | NOT Available (occurs later) |
| ---------- | -------------------------------- | ---------------------------- |
| **SQ**     | FP1 only, historical             | S, Q, R                      |
| **S**      | FP1, SQ grid, historical         | Q, R                         |
| **Q**      | FP1, SQ, S, historical           | R                            |
| **R**      | FP1, Q grid, SQ, S, historical   | -                            |

Note: Sprint weekends have NO FP2 or FP3 sessions.

### Key Modules

**Feature Pipelines** (`src/features/`):

- `base_pipeline.py` - Abstract base class for all pipelines
- `qualifying_pipeline.py` - Qualifying features (FP1-3 available)
- `sprint_quali_pipeline.py` - Sprint qualifying (FP1 only)
- `sprint_race_pipeline.py` - Sprint race (FP1 + SQ grid)
- `race_pipeline.py` - Race features (all sessions + Q grid)
- `grid_features.py` - Grid position feature extractor

**Feature Extractors** (`src/features/`) - all use `.shift(1)` before `.rolling()`:

- `sector_features.py` - Sector time analysis (S1/S2/S3 strengths)
- `qualifying_features.py` - Q1→Q2→Q3 progression patterns
- `practice_features.py` - FP1/FP2/FP3 pace correlation
- `tyre_features.py` - Compound preferences and performance
- `sprint_features.py` - Sprint qualifying and sprint race features
- `weather_features.py` - Track/air temp, humidity, wet conditions
- `elo_features.py` - Driver and constructor ELO ratings
- `reliability_features.py` - DNF rates and driver reliability
- `first_lap_features.py` - Lap 1 position change performance
- `momentum_features.py` - Recent form and trend features
- `relative_features.py` - Position vs field and teammate deltas
- `circuit_features.py` - Circuit-specific driver/team performance
- `circuit_overtaking_features.py` - Overtaking rates at specific circuits
- `driver_circuit_features.py` - Driver-circuit interaction patterns
- `track_evolution_features.py` - Track grip improvement throughout session
- `wet_weather_skill_features.py` - Wet weather performance analysis
- `race_pace_features.py` - Position changes and race pace progression

**Models** (`src/models/`):

- `base_ranker.py` - Base LGBMRanker class with shared logic
- `qualifying.py` - QualifyingLGBMRanker
- `race.py` - RaceLGBMRanker
- `sprint_qualifying.py` - SprintQualiLGBMRanker (higher regularization)
- `sprint_race.py` - SprintRaceLGBMRanker (higher regularization)
- `baselines.py` - Baseline methods for comparison (FP3, Grid, Championship, etc.)

**Data** (`src/data/`):

- `fastf1_sync.py` - Syncs F1 data from FastF1 API to parquet format
- `loaders.py` - Loads session parquet files with temporal filtering

**Evaluation**:

- `src/evaluation/scoring.py` - Implements the 2/1/0 point scoring system

**CLI Tools** (`src/cli/`):

- `retrain.py` - Train models with cross-validation
- `predict.py` - Generate predictions for upcoming races
- `evaluate.py` - Evaluate model performance
- `tune_hyperparams.py` - Optuna hyperparameter optimization
- `export_onnx.py` - Export models to ONNX format (patches lambdarank→regression, see [onnxmltools#338](https://github.com/onnx/onnxmltools/issues/338))
- `export_features.py` - Export computed features to JSON for cross-platform inference
- `backtest.py` - Run temporal backtests across a full season

**PHP Inference** (`inference/php/`):

- `predict.php` - Single-file ONNX inference script
- `Dockerfile` - Builds PHP 8.5 with ORT extension
- Run with: `make inference/php FEATURES=<file.json>`

Data flow: FastF1 API → parquet files → feature pipeline → model training → predictions

For cross-platform inference: pickle models → ONNX export → feature JSON export → PHP/C++/Rust via ONNX Runtime

## Code Style

- Python 3.11+ with type hints on all public functions
- Google-style docstrings
- 100 char line length (enforced by ruff)
- TimeSeriesSplit for cross-validation (never random splits)

## Key Constraints

- All features must pass temporal leakage tests (use `.shift(1)` before `.rolling()`)
- New drivers/circuits fall back to team-based or circuit-type predictions
