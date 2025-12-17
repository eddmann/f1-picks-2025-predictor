# F1 Picks 2025 Predictor

Machine learning system for predicting Formula 1 top-3 finishers using LightGBM LambdaRank models. Built for our family [F1 Picks prediction game](https://github.com/eddmann/f1-picks-2025).

## The Game

The game uses a simple points system for predicting podium positions:

| Prediction                                         | Points |
| -------------------------------------------------- | ------ |
| Correct driver in correct position (P1, P2, or P3) | 2      |
| Correct driver in wrong position                   | 1      |
| Driver not in actual top-3                         | 0      |

Maximum per session: 6 points (all three drivers in exact positions)

The system predicts four session types (availability varies by race weekend):

- **Qualifying (Q)** - Predict qualifying top-3
- **Race (R)** - Predict race top-3
- **Sprint Qualifying (SQ)** - Predict sprint qualifying top-3 (sprint weekends only)
- **Sprint Race (S)** - Predict sprint race top-3 (sprint weekends only)

## The Challenge

The goal of this project is to build an ML model that outperforms the simple prediction strategies most people use:

| Baseline    | Strategy                                                                      | Rationale                                                               |
| ----------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| **FP3/SQ**  | Predict qualifying based on FP3 fastest laps (or SQ times on sprint weekends) | Final practice before qualifying where teams run qualifying simulations |
| **Q Grid**  | Predict race results based on qualifying order                                | Starting position strongly correlates with finishing position in F1     |
| **FP1**     | Predict sprint qualifying based on FP1 fastest laps                           | Only practice session available before sprint qualifying                |
| **SQ Grid** | Predict sprint race based on sprint qualifying grid                           | Grid position predicts sprint race finish                               |

These baselines are surprisingly effective in F1 because:

- Limited overtaking - unlike many motorsports, F1 races often see few position changes
- Qualifying simulations - teams run representative pace in FP3, making it predictive of qualifying
- Short sprints - sprint races have little time for strategy to play out

Can machine learning do better by incorporating historical performance, driver form, ELO ratings, and circuit-specific patterns?

## Getting Started

### Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- ~1GB disk space for F1 data

### Install & Run

```bash
# Clone and install
git clone https://github.com/eddmann/f1-picks-2025-predictor.git
cd f1-picks-2025-predictor
make install

# Download historical data (2020-2025)
make data

# Train all models
make train

# Generate predictions
make predict/qualifying RACE=2025-24
make predict/race RACE=2025-24

# Run backtest
make backtest SEASON=2025
```

See `make help` for all available commands.

### Race Weekend Workflow

#### Standard Weekend

| When              | Command                                |
| ----------------- | -------------------------------------- |
| After FP3         | `make data/sync SEASON=2025 ROUND=24`  |
| Before Qualifying | `make predict/qualifying RACE=2025-24` |
| After Qualifying  | `make data/sync SEASON=2025 ROUND=24`  |
| Before Race       | `make predict/race RACE=2025-24`       |

#### Sprint Weekend

Sprint weekends have no FP2/FP3 - just FP1 on Friday, then Sprint Qualifying, Sprint Race, Qualifying, and Race.

| When                     | Command                                  |
| ------------------------ | ---------------------------------------- |
| After FP1                | `make data/sync SEASON=2025 ROUND=24`    |
| Before Sprint Qualifying | `make predict/sprint_quali RACE=2025-24` |
| After Sprint Qualifying  | `make data/sync SEASON=2025 ROUND=24`    |
| Before Sprint Race       | `make predict/sprint_race RACE=2025-24`  |
| After Sprint Race        | `make data/sync SEASON=2025 ROUND=24`    |
| Before Qualifying        | `make predict/qualifying RACE=2025-24`   |
| After Qualifying         | `make data/sync SEASON=2025 ROUND=24`    |
| Before Race              | `make predict/race RACE=2025-24`         |

## How It Works

### Model Architecture

Each session type has its own **LightGBM LambdaRank** model optimised for learning-to-rank. The models predict a relevance score for each driver, then rank them to get the top-3.

#### Why LambdaRank?

Traditional classification (predicting P1/P2/P3 as classes) doesn't capture that P2 is "closer" to P1 than P10 is. LambdaRank optimises pairwise rankings directly.

### Features

The model uses ~300 features across several categories:

#### ELO Rating System

Driver and constructor ELO ratings updated after each race, providing a dynamic measure of relative performance:

- Individual driver ELO with K-factor of 32
- Constructor ELO with K-factor of 24
- ELO percentile rankings within the field

#### Rolling Performance Metrics

Historical performance aggregated over multiple windows (3, 5, 10 races):

- Average finishing position
- Top-3 finish rate
- Position gained/lost versus grid
- Teammate performance delta

#### Practice Session Analysis

- Sector time analysis (S1/S2/S3 strengths)
- Lap time percentiles within session
- Long run pace on race weekend
- Tyre compound performance patterns

#### Contextual Features

- Circuit characteristics and historical performance
- Weather conditions (temperature, humidity, wet sessions)
- First lap position change tendencies
- Reliability/DNF rates

### Temporal Integrity

All features use strict temporal ordering - predictions for Race N only use data from Races 1 to N-1. This prevents data leakage and ensures realistic evaluation.

### Data Availability by Session

The 2025 sprint format is: FP1 → Sprint Qualifying → Sprint Race → Qualifying → Race

| Predicting  | Available Data                                       | NOT Available        |
| ----------- | ---------------------------------------------------- | -------------------- |
| SQ          | FP1, historical                                      | Sprint Race, Q, Race |
| Sprint Race | FP1, SQ grid, historical                             | Q, Race              |
| Qualifying  | FP1/FP2/FP3 (or SQ/S on sprint weekends), historical | Race                 |
| Race        | All sessions + Q grid                                | -                    |

### Hyperparameter Tuning

Models are tuned using **Optuna** with Bayesian optimisation. Key parameters include boosting rounds, tree complexity, learning rate, and regularisation strength. Run tuning with:

```bash
make train/tune TYPE=qualifying TRIALS=50
```

## 2025 Season Backtest

Having built this model late in the 2025 season, I wanted to evaluate how it would have performed across the entire year. The backtest uses strict temporal validation: for each race, a fresh model is trained using only data that would have been available before that race, preventing any data leakage.

### Results Summary

| Session     | Sessions | Model Avg | Model Total | Baseline      | Baseline Total | Delta     |
| ----------- | -------- | --------- | ----------- | ------------- | -------------- | --------- |
| Qualifying  | 24       | 1.92      | 46/144      | FP3/SQ: 2.67  | 64/144         | **-0.75** |
| Race        | 24       | 1.88      | 45/144      | Q Grid: 3.67  | 88/144         | **-1.79** |
| SQ          | 6        | 1.33      | 8/36        | FP1: 3.00     | 18/36          | **-1.67** |
| Sprint Race | 6        | 0.83      | 5/36        | SQ Grid: 3.83 | 23/36          | **-3.00** |
| **Total**   | **60**   | **1.73**  | **104/360** | -             | **193/360**    | **-89**   |

### Analysis

The model underperforms baselines across all session types, with baselines scoring 89 more points over the season. Several factors contribute:

1. Grid position dominance - the Q Grid baseline achieves 3.67 pts/race by simply predicting qualifying order holds. The model attempts to predict position changes but the variance hurts more than it helps.

2. FP3 reliability - teams' qualifying simulation runs in FP3 closely predict actual qualifying, giving the FP3 baseline 2.67 pts/session with no complexity.

3. Limited sprint training data - sprint weekends began in 2021 with few events per year. Sprint models train on ~160-240 samples versus 1500+ for qualifying/race models.

4. Early season cold start - with sparse current-year data, early predictions suffer:

   | Rounds  | Model Avg |
   | ------- | --------- |
   | R1-R6   | 0.67      |
   | R7-R12  | 1.83      |
   | R13-R18 | 2.42      |
   | R19-R24 | 2.50      |

### Where the Model Adds Value

The model outperformed baselines in specific scenarios:

| Race            | Session | Model | Baseline | Delta  |
| --------------- | ------- | ----- | -------- | ------ |
| R10 Montreal    | Q       | 4     | 1        | **+3** |
| R24 Abu Dhabi   | R       | 6     | 4        | **+2** |
| R5 Jeddah       | R       | 3     | 2        | **+1** |
| R12 Silverstone | R       | 4     | 3        | **+1** |
| R14 Hungary     | Q       | 4     | 3        | **+1** |
| R14 Hungary     | R       | 4     | 3        | **+1** |
| R16 Monza       | Q       | 4     | 3        | **+1** |
| R17 Baku        | Q       | 2     | 1        | **+1** |
| R24 Abu Dhabi   | Q       | 4     | 3        | **+1** |

The model tends to help when practice sessions are unrepresentative (weather, red flags) or when significant position changes occur.

### Race-by-Race Results

<details>
<summary>Full breakdown</summary>

| Round | Circuit     | Q     | Q Base | R     | R Base | SQ  | SQ Base | S   | S Base |
| ----- | ----------- | ----- | ------ | ----- | ------ | --- | ------- | --- | ------ |
| R1    | Melbourne   | 2     | 3      | 0     | 3      | -   | -       | -   | -      |
| R2    | Shanghai    | 0     | 1      | 0     | 4      | 0   | 2       | 0   | 4      |
| R3    | Suzuka      | 1     | 2      | 0     | 6      | -   | -       | -   | -      |
| R4    | Sakhir      | 1     | 4      | 0     | 4      | -   | -       | -   | -      |
| R5    | Jeddah      | 2     | 4      | **3** | 2      | -   | -       | -   | -      |
| R6    | Miami       | 0     | 2      | 1     | 2      | 0   | 1       | 0   | 3      |
| R7    | Imola       | 0     | 2      | 0     | 2      | -   | -       | -   | -      |
| R8    | Monaco      | 1     | 2      | 2     | 6      | -   | -       | -   | -      |
| R9    | Barcelona   | 2     | 4      | 4     | 4      | -   | -       | -   | -      |
| R10   | Montreal    | **4** | 1      | 1     | 4      | -   | -       | -   | -      |
| R11   | Spielberg   | 2     | 3      | 1     | 4      | -   | -       | -   | -      |
| R12   | Silverstone | 2     | 3      | **4** | 3      | -   | -       | -   | -      |
| R13   | Spa         | 3     | 2      | 2     | 4      | 0   | 6       | 0   | 4      |
| R14   | Budapest    | **4** | 3      | **4** | 3      | -   | -       | -   | -      |
| R15   | Zandvoort   | 2     | 2      | 1     | 3      | -   | -       | -   | -      |
| R16   | Monza       | **4** | 3      | 4     | 6      | -   | -       | -   | -      |
| R17   | Baku        | **2** | 1      | 2     | 3      | -   | -       | -   | -      |
| R18   | Singapore   | 2     | 3      | 2     | 4      | -   | -       | -   | -      |
| R19   | Austin      | 1     | 4      | 1     | 6      | 3   | 3       | 2   | 2      |
| R20   | Mexico City | 1     | 3      | 2     | 4      | -   | -       | -   | -      |
| R21   | Sao Paulo   | 3     | 4      | 2     | 4      | 3   | 3       | 1   | 4      |
| R22   | Las Vegas   | 1     | 2      | 1     | 1      | -   | -       | -   | -      |
| R23   | Qatar       | 2     | 3      | 2     | 2      | 2   | 3       | 2   | 6      |
| R24   | Abu Dhabi   | **4** | 3      | **6** | 4      | -   | -       | -   | -      |

**Bold** = Model beat baseline

</details>

## Model Export (ONNX)

Models can be exported to ONNX format for cross-platform inference:

```bash
# Export all models to ONNX
make export/onnx

# Export specific model
make export/onnx/qualifying

# Export features to JSON for inference
make export/features RACE=2025-24 TYPE=qualifying
```

The ONNX export enables inference in PHP, C++, Rust, or any language with ONNX Runtime support. See [inference/php/](inference/php/) for a PHP implementation example.

Note: LambdaRank models require a patch during export due to [onnxmltools#338](https://github.com/onnx/onnxmltools/issues/338).

## Acknowledgements

- [FastF1](https://github.com/theOehrly/Fast-F1) for the F1 timing data API
- [LightGBM](https://github.com/microsoft/LightGBM) for the gradient boosting framework
- [F1 ELO by Matthew Perron](https://matthewperron.github.io/f1-elo/) for the ELO methodology
- [F1-Predictor](https://github.com/JaideepGuntupalli/f1-predictor) for feature engineering ideas

## License

MIT License - see [LICENSE](LICENSE) file.
