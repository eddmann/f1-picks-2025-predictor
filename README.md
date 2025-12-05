# F1 Picks ML Prediction System

## Why This Exists

An exploration into using ML to predict the top-3 finishers for my family's F1 picks game, covering all session types including sprints:

- **2 points** for a correct driver in the correct position
- **1 point** for a correct driver in the wrong position
- **0 points** if the driver isn't in the top-3
- **Max 6 points** per session (3 drivers × 2 points)

**The challenge:** Can machine learning beat the simple strategies anyone would use?

| Instead of ML, you could just... | Called                |
| -------------------------------- | --------------------- |
| Pick whoever's fastest in FP3    | FP3 baseline          |
| Pick the championship top-3      | Championship baseline |
| Assume grid order = finish order | Grid baseline         |

If ML can't beat these trivial approaches, it provides no value.

## Results

| Model            | CV Score | Key Insight                                  |
| ---------------- | -------- | -------------------------------------------- |
| **Qualifying**   | 3.18 pts | ELO ratings + sector times + practice pace   |
| **Race**         | 2.80 pts | Grid position explains ~60% of race outcome  |
| **Sprint Quali** | 2.40 pts | Limited data (~6 weekends/year since 2021)   |
| **Sprint Race**  | 2.90 pts | Grid position (from SQ) is primary predictor |

**Bottom line:** The models provide useful predictions, but race outcomes are fundamentally constrained by grid position dominance - cars starting at the front usually finish there.

## The Journey

How this project evolved - experiments, failures, and insights.

**[Read the full development story →](docs/journey.md)**

Highlights:

- **Phase 1:** Kaggle → FastF1, Classification → LambdaRank
- **Phase 2:** Baseline reality check (sobering truths)
- **Phase 3:** ELO ratings (+3%), Reliability (+2%), First lap features (+6.5%)
- **Phase 4:** The race prediction wall (grid dominates, chaos explains the rest)
- **Phase 5:** Hyperparameter tuning (+7.6%)
- **Phase 6:** Wet weather, track evolution, circuit overtaking features (+7.4%)

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
  P1: PIA (McLaren) - score: 5.156
  P2: NOR (McLaren) - score: 3.831
  P3: VER (Red Bull Racing) - score: 2.239
============================================================
```

## Technical Reference

Detailed documentation on features, model architecture, and training.

**[Read the technical reference →](docs/technical-reference.md)**

Topics covered:

- Model architecture (LightGBM LambdaRank)
- Feature categories and importance
- ELO rating system (10 features)
- Reliability features (24 features)
- First lap features (35 features)
- Temporal integrity patterns
- Training options and hyperparameters
- Model export formats

## License

MIT License - see [LICENSE](LICENSE) file.

Note: This project depends on [FastF1](https://github.com/theOehrly/Fast-F1) (GPL-3.0). If you redistribute with FastF1 included, you must comply with GPL-3.0 terms.

## Acknowledgments

- [FastF1](https://github.com/theOehrly/Fast-F1) - F1 timing data access
- [F1 ELO by Matthew Perron](https://matthewperron.github.io/f1-elo/) - ELO methodology
- [F1-Predictor](https://github.com/JaideepGuntupalli/f1-predictor) - Feature engineering ideas
