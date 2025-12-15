# F1 Picks 2025 Predictor

An exploration into using ML to predict the top-3 finishers for my family's F1 picks game, covering all session types including sprints:

- **2 points** for a correct driver in the correct position
- **1 point** for a correct driver in the wrong position
- **0 points** if the driver isn't in the top-3
- **Max 6 points** per session (3 drivers × 2 points)

**The challenge:** Can machine learning beat simple strategies anyone would use - like picking whoever's fastest in practice, or assuming the grid order equals finish order?

| Session      | Baseline (what anyone would pick)            |
| ------------ | -------------------------------------------- |
| Qualifying   | FP3 fastest (or SQ top-3 on sprint weekends) |
| Race         | Grid position from qualifying                |
| Sprint Quali | FP1 fastest                                  |
| Sprint Race  | SQ grid position                             |

If ML can't beat these trivial approaches, it provides no value.

## Results

Cross-validated on 5-fold temporal splits (2020-2025 data):

| Model            | CV Score     | Accuracy | Key Insight                                 |
| ---------------- | ------------ | -------- | ------------------------------------------- |
| **Qualifying**   | 3.34/6.0 pts | 56%      | ELO ratings + practice pace + sector times  |
| **Race**         | 3.18/6.0 pts | 53%      | Grid position explains ~60% of race outcome |
| **Sprint Quali** | 4.00/6.0 pts | 67%      | Historical quali correlation + ELO ratings  |
| **Sprint Race**  | 3.60/6.0 pts | 60%      | SQ grid position + ELO ratings              |

**Season projection:** ~202 points out of 360 possible (56%) across all sessions.

**What this means in practice:**

- Typically predicts 1-2 exact position matches per session
- Gets the right drivers in top-3 ~55-65% of the time
- Sprint Qualifying is the strongest model despite limited data (11 sessions)

**Bottom line:** The models provide useful predictions that beat simple baselines. Race outcomes remain hardest to predict due to strategy variance, incidents, and the fundamental constraint that cars starting at the front usually finish there.

## Quick Start

### Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- ~1GB disk space for F1 data

### Install & Setup

```bash
# Clone and install
git clone https://github.com/eddmann/f1-picks-2025-predictor.git
cd f1-picks-2025-predictor
make install

# Sync F1 data
make data/sync/2025

# Train models
make train

# Generate predictions
make predict/qualifying RACE=2025-qatar
```

### Race Weekend Workflow

**Standard Weekend (Fri-Sun):**

| When              | What to do                                       |
| ----------------- | ------------------------------------------------ |
| After FP3         | `make data/sync SEASON=2025 ROUND=24`            |
| Before Qualifying | `make predict/qualifying RACE=2025-24`           |
| After Qualifying  | `make data/sync SEASON=2025 ROUND=24` (get grid) |
| Before Race       | `make predict/race RACE=2025-24`                 |

**Sprint Weekend (Fri-Sun):**

Sprint weekends have no FP2/FP3 - just FP1 on Friday.

| Day      | When                | What to do                               |
| -------- | ------------------- | ---------------------------------------- |
| Friday   | After FP1           | `make data/sync SEASON=2025 ROUND=24`    |
| Friday   | Before Sprint Quali | `make predict/sprint_quali RACE=2025-24` |
| Friday   | After Sprint Quali  | `make data/sync SEASON=2025 ROUND=24`    |
| Saturday | Before Sprint Race  | `make predict/sprint_race RACE=2025-24`  |
| Saturday | After Sprint Race   | `make data/sync SEASON=2025 ROUND=24`    |
| Saturday | Before Qualifying   | `make predict/qualifying RACE=2025-24`   |
| Saturday | After Qualifying    | `make data/sync SEASON=2025 ROUND=24`    |
| Sunday   | Before Race         | `make predict/race RACE=2025-24`         |

**With SHAP explanation (understand why):**

```bash
make predict/explain RACE=2025-24 TYPE=qualifying
```

### Example Output

```
============================================================
F1 QUALIFYING PREDICTION
============================================================
Race: Abu Dhabi Grand Prix (Yas Island)
Year: 2025, Round: 24

PREDICTED TOP 3:
  P1: NOR (McLaren) - score: 9.063
  P2: PIA (McLaren) - score: 8.432
  P3: VER (Red Bull Racing) - score: 7.056
============================================================
```

## How It Works

### Model Architecture

Each session type has its own **LightGBM LambdaRank** model optimized for learning-to-rank. The models predict a relevance score for each driver, then rank them to get the top-3.

**Why LambdaRank?** Traditional classification (predicting P1/P2/P3 as classes) doesn't capture that P2 is "closer" to P1 than P10 is. LambdaRank optimizes pairwise rankings directly.

### Feature Categories

~250-330 features per model (varies by session type):

| Category            | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| **ELO Ratings**     | Driver and constructor skill ratings updated after each race |
| **Recent Form**     | Rolling averages of positions, top-3 rates, consistency      |
| **Practice Pace**   | FP1/FP2/FP3 lap times, gaps to leader, improvement trends    |
| **Circuit History** | Driver performance at specific tracks and circuit types      |
| **Reliability**     | DNF rates, mechanical failures, incident patterns            |
| **First Lap**       | Position gains/losses on lap 1, overtaking ability           |
| **Weather**         | Wet weather skill, temperature preferences                   |
| **Grid Position**   | Qualifying result features (race models only)                |
| **Sector Times**    | S1/S2/S3 strengths, theoretical best laps                    |
| **Track Evolution** | Grip improvement through sessions                            |

### Temporal Integrity

All features use **strict temporal ordering** - predictions for Race N only use data from Races 1 to N-1. This prevents data leakage and ensures realistic evaluation.

**Sprint Weekend Constraints:**

The 2025 sprint format is: FP1 → Sprint Quali → Sprint Race → Qualifying → Race

| Predicting   | Available Data              | NOT Available        |
| ------------ | --------------------------- | -------------------- |
| Sprint Quali | FP1, historical             | Sprint Race, Q, Race |
| Sprint Race  | FP1, SQ grid, historical    | Q, Race              |
| Qualifying   | FP1, SQ, Sprint, historical | Race                 |
| Race         | All sessions + Q grid       | -                    |

### Hyperparameter Tuning

Models are tuned using **Optuna** with Bayesian optimization over 50 trials. Key parameters:

- `n_estimators`: 80-300 boosting rounds
- `num_leaves`: 15-50 tree complexity
- `learning_rate`: 0.01-0.2
- `min_child_samples`: regularization strength
- `subsample` / `colsample_bytree`: feature/row sampling

Sprint models benefited most from tuning (+67% for Sprint Quali) as the original conservative settings were too restrictive.

## Development

### Training & Tuning

```bash
# Train all models with current hyperparameters
make train

# Hyperparameter tuning (50 Optuna trials)
make train/tune TYPE=qualifying TRIALS=50
make train/tune TYPE=race TRIALS=50
make train/tune TYPE=sprint_quali TRIALS=50
make train/tune TYPE=sprint_race TRIALS=50
```

### Testing

```bash
make test          # Run all tests
make test/unit     # Unit tests only
make lint          # Run ruff linter
make ci            # Full CI checks (lint + format + test)
```

### Data Management

```bash
make data/status   # Show data summary
make data/sessions # List available sessions
make data/sync SEASON=2025 ROUND=5  # Sync specific round
```

### Model Export (ONNX)

Export models to ONNX format for cross-platform inference:

```bash
make export/onnx              # Export all models
make export/features RACE=2025-24 TYPE=qualifying  # Export features for a race
```

See [`inference/php/`](inference/php/) for running predictions via PHP + ONNX Runtime.

**Note:** Models use LightGBM's `lambdarank` objective, which is patched to `regression` for ONNX compatibility. Predictions remain identical.

## License

MIT License - see [LICENSE](LICENSE) file.

Note: This project depends on [FastF1](https://github.com/theOehrly/Fast-F1) (GPL-3.0). If you redistribute with FastF1 included, you must comply with GPL-3.0 terms.

## Acknowledgments

- [FastF1](https://github.com/theOehrly/Fast-F1) - F1 timing data access
- [F1 ELO by Matthew Perron](https://matthewperron.github.io/f1-elo/) - ELO methodology
- [F1-Predictor](https://github.com/JaideepGuntupalli/f1-predictor) - Feature engineering ideas
