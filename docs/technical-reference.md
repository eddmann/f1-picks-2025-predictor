# Technical Reference

Detailed technical documentation for the F1 prediction system.

## Model Architecture

The system uses **LightGBM LambdaRank** - a learning-to-rank model that directly optimizes driver ordering within each session. Unlike classification models that predict "will this driver finish top-3?", the ranker learns to compare drivers against each other.

### Why Learning-to-Rank?

| Approach       | Output                           | Problem                                     |
| -------------- | -------------------------------- | ------------------------------------------- |
| Classification | "70% chance of top-3"            | Doesn't rank drivers relative to each other |
| Regression     | "Expected position: 4.2"         | Hard to convert to rankings                 |
| **LambdaRank** | "Driver A > Driver B > Driver C" | Directly optimizes ranking                  |

### Session-Specific Models

Each prediction type has its own model due to different data availability:

| Model                 | Available Data            | Features |
| --------------------- | ------------------------- | -------- |
| **Qualifying**        | FP1, FP2, FP3, historical | ~299     |
| **Race**              | All practice + Q grid     | ~304     |
| **Sprint Qualifying** | FP1 only, historical      | ~254     |
| **Sprint Race**       | FP1 + SQ grid             | ~259     |

**Note:** Sprint weekends have no FP2/FP3 - only FP1 is available before Sprint Qualifying.

---

## Feature Categories

Features are grouped by type, with importance percentages from trained models:

| Category                 | Qualifying | Race      | Description                                   |
| ------------------------ | ---------- | --------- | --------------------------------------------- |
| **Relative Performance** | 33.5%      | 11.2%     | Position vs field, teammate deltas            |
| **Rolling Position**     | 15.8%      | 6.4%      | Historical position averages (3/5/10 windows) |
| **Practice Times**       | 14.2%      | 11.8%     | FP1/FP2/FP3 rankings and gaps                 |
| **Grid Position**        | -          | **52.0%** | Qualifying position (race only)               |
| **Momentum**             | 8.7%       | 2.8%      | EWM position, form trends                     |
| **Circuit History**      | 8.4%       | 5.8%      | Driver's history at specific circuit          |
| **ELO Ratings**          | 7.8%       | 3.1%      | Driver/constructor skill ratings              |
| **Sector Times**         | 7.2%       | 0%        | S1/S2/S3 historical times                     |
| **Team Performance**     | 3.2%       | 2.4%      | Constructor rolling averages                  |
| **Reliability/DNF**      | 1.2%       | 4.5%      | DNF rates, driver confidence                  |

---

## ELO Rating Features

Inspired by chess ELO ratings. Key insight: **You can't compare drivers across teams** (different cars). Solution: only compare teammates.

### Features (10 total)

| Feature                      | Description                                               |
| ---------------------------- | --------------------------------------------------------- |
| `driver_elo`                 | Driver skill rating from teammate comparisons             |
| `constructor_elo`            | Team/car performance from field comparisons               |
| `ga_driver_elo`              | Grid-adjusted driver ELO (accounts for starting position) |
| `combined_elo`               | 30% driver + 70% constructor (car matters more)           |
| `driver_elo_percentile`      | Driver's ELO rank within the session field                |
| `constructor_elo_percentile` | Constructor's ELO rank within the session field           |
| `driver_elo_vs_avg`          | How much driver's ELO differs from session average        |
| `constructor_elo_vs_avg`     | How much constructor's ELO differs from session average   |
| `driver_elo_vs_initial`      | ELO change from starting rating (1500)                    |
| `constructor_elo_vs_initial` | Constructor ELO change from starting rating               |

### How ELO Works

1. **Driver ELO** - For each session, compare teammates:

   - Winner gains ELO points, loser loses points
   - Amount depends on expected outcome (upset = bigger swing)
   - K-factor: 32 (standard chess value)

2. **Constructor ELO** - Compare team's best result against field:

   - Round-robin comparison against all other teams
   - Captures relative car competitiveness
   - K-factor: 24 (more stable than driver ratings)

3. **Grid-Adjusted ELO** - Adjusts driver ELO based on grid position:
   - P1 gets +47.5 ELO adjustment
   - P10 gets +2.5
   - P20 gets -47.5
   - Helps predict who might outperform their grid slot

### Impact

| Model      | Before ELO | With ELO | Improvement |
| ---------- | ---------- | -------- | ----------- |
| Qualifying | 2.56 pts   | 2.62 pts | +2.3%       |
| Race       | 2.78 pts   | 2.88 pts | +3.6%       |

---

## Reliability Features

DNF rates and driver/team completion patterns.

### Features (24 total)

| Feature                          | Description                        |
| -------------------------------- | ---------------------------------- |
| `driver_dnf_rate_{w}`            | Rolling DNF rate over last w races |
| `driver_mechanical_dnf_rate_{w}` | Rate of mechanical failures        |
| `driver_incident_dnf_rate_{w}`   | Rate of crashes/collisions         |
| `driver_finish_rate_{w}`         | Completion rate (1 - DNF rate)     |
| `driver_confidence`              | Career completion percentage       |
| `driver_season_confidence`       | Season completion percentage       |
| `races_since_dnf`                | Consecutive race finishes          |
| `dnf_last_race`                  | Binary: did driver DNF last race?  |
| `driver_dnf_vs_field`            | Driver's DNF rate vs field average |
| `team_mechanical_dnf_rate_{w}`   | Team's mechanical failure rate     |
| `team_career_dnf_rate`           | Team's overall reliability         |

### DNF Classification

- **Mechanical**: Engine, gearbox, suspension, power unit, fuel, cooling
- **Incident**: Accident, collision, spun off, crash
- **Other**: Disqualified, illness, did not start

### Impact

| Model      | Before DNF | With DNF | Improvement |
| ---------- | ---------- | -------- | ----------- |
| Qualifying | 2.62 pts   | 2.62 pts | 0%          |
| Race       | 2.88 pts   | 2.94 pts | +2.2%       |

DNF features primarily help race prediction where reliability matters over 50+ laps.

---

## First Lap Features

Position changes from race lap 1, applied to Qualifying/Sprint only.

### Features (35 total)

| Feature                       | Description                                      |
| ----------------------------- | ------------------------------------------------ |
| `first_lap_avg_gain_{w}`      | Avg positions gained on lap 1 over w races       |
| `first_lap_gain_rate_{w}`     | % of races where driver gained positions         |
| `first_lap_loss_rate_{w}`     | % of races where driver lost positions           |
| `first_lap_career_avg_gain`   | Career average lap 1 position change             |
| `start_specialist`            | Binary: consistently gains 1+ positions on lap 1 |
| `start_conservative`          | Binary: consistently loses positions on lap 1    |
| `team_first_lap_avg_gain_{w}` | Team's average lap 1 performance                 |

### Data Limitation

FastF1's lap-level `position` column contains **final race position repeated**, not actual lap-by-lap positions. This means:

- True lap 1 position changes cannot be computed directly
- For qualifying, these features provide new signal
- For race, they duplicate existing features (not applied)

### Impact

| Model      | Before   | With         | Improvement |
| ---------- | -------- | ------------ | ----------- |
| Qualifying | 2.62 pts | **2.79 pts** | **+6.5%**   |
| Race       | 2.94 pts | 3.04 pts     | +3.4%       |

---

## Most Predictive Features

### Qualifying Model (Top 10)

1. `ewm_position_5` - Exponentially weighted recent form
2. `avg_pos_vs_field_10` - Position relative to field over 10 races
3. `avg_percentile_10` - Historical percentile ranking
4. `ga_driver_elo` - Grid-adjusted driver ELO rating
5. `current_fp3_rank` - FP3 ranking (most predictive practice)
6. `driver_elo` - Driver skill rating
7. `circuit_type_top3_rate` - Top-3 rate at this circuit type
8. `avg_pos_vs_field_5` - Position relative to field over 5 races
9. `rolling_s1_rank_3` - Recent sector 1 performance
10. `current_fp3_gap_pct` - Gap to fastest in FP3

### Race Model (Top 5)

1. `current_grid_percentile` (42.6%) - Grid position relative to field
2. `current_grid_top5` (6.1%) - Binary: started in top 5
3. `current_grid_position` (5.8%) - Raw grid position
4. `ewm_position_3` (3.4%) - Recent form
5. `ewm_position_5` (2.2%) - Recent form over 5 races

**Grid features account for 55% of race model importance.**

---

## Feature Pruning

Two-pass training removes zero-importance features automatically.

### Results

| Model        | Before       | After        | Score Change        |
| ------------ | ------------ | ------------ | ------------------- |
| Qualifying   | 174 features | 159 features | 2.48 → 2.65 (+6.9%) |
| Race         | 194 features | 112 features | 2.97 → 3.05 (+2.7%) |
| Sprint Race  | 106 features | 47 features  | No change           |
| Sprint Quali | 101 features | 15 features  | No change           |

### Zero-Importance Categories (Race Model)

- Sector features (31) - Single-lap metrics don't predict race
- Sprint features (15) - Only relevant for sprint weekends
- Tyre features (14) - Qualifying tyre usage ≠ race outcome
- Weather features (8) - Conditions at quali ≠ conditions at race

---

## Temporal Integrity

All features use `.shift(1).rolling()` patterns to ensure:

- Features for Race N only use data from Races 1 to N-1
- No data leakage from the session being predicted
- Practice session data is from the same weekend (available before qualifying)

```python
# CORRECT: shift(1) before rolling prevents leakage
df['rolling_avg'] = df.groupby('driver')['position'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()
)

# WRONG: rolling without shift uses current race data
df['rolling_avg'] = df.groupby('driver')['position'].transform(
    lambda x: x.rolling(5, min_periods=1).mean()
)
```

---

## Training Options

```bash
# Train all models with feature pruning (recommended)
uv run python -m src.cli.retrain --type all --prune-features

# Train specific model type
uv run python -m src.cli.retrain --type qualifying
uv run python -m src.cli.retrain --type race

# Train with hyperparameter tuning (slower)
uv run python -m src.cli.retrain --type qualifying --tune --tune-trials 50
```

### Hyperparameters

**Default LightGBM parameters:**

- `n_estimators`: 100
- `num_leaves`: 31
- `learning_rate`: 0.1
- `min_child_samples`: 20

**Tuned Qualifying parameters:**

- `n_estimators`: 259
- `num_leaves`: 15
- `learning_rate`: 0.082

**Tuned Race parameters:**

- `n_estimators`: 192
- `num_leaves`: 20
- `learning_rate`: 0.133
- `reg_lambda`: 0.326

---

## Model Export Formats

Models are saved in two formats:

| Format              | File              | Use Case                          |
| ------------------- | ----------------- | --------------------------------- |
| **Pickle**          | `*.pkl`           | Python with scikit-learn/LightGBM |
| **LightGBM Native** | `*.lgbm.txt`      | Cross-platform deployment         |
| **Feature Names**   | `*.features.json` | Feature alignment                 |

### Loading in Python

```python
from src.models.qualifying import load_model

model = load_model("models/qualifying_ranker.pkl")
predictions = model.predict(features)
```

### Loading LightGBM Native Format

```python
import lightgbm as lgb
import json

# Load model
booster = lgb.Booster(model_file="models/qualifying_ranker.lgbm.txt")

# Load feature names
with open("models/qualifying_ranker.lgbm.features.json") as f:
    feature_names = json.load(f)["features"]

# Predict
predictions = booster.predict(features[feature_names])
```

---

## Scoring System

The prediction game uses:

- **2 points**: Correct driver in correct position
- **1 point**: Correct driver in wrong position
- **0 points**: Driver not in top-3
- **Maximum**: 6 points per session (3 positions × 2 points)

### Example

| Predicted     | Actual        | Points            |
| ------------- | ------------- | ----------------- |
| VER, NOR, LEC | VER, LEC, NOR | 2 + 1 + 1 = 4 pts |
| VER, HAM, RUS | VER, NOR, LEC | 2 + 0 + 0 = 2 pts |
| NOR, VER, LEC | VER, NOR, LEC | 1 + 1 + 2 = 4 pts |

---

## References

- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [F1 ELO by Matthew Perron](https://matthewperron.github.io/f1-elo/)
- [F1 Rating System](https://github.com/mwtmurphy/f1-elo)
- [F1-Predictor](https://github.com/JaideepGuntupalli/f1-predictor)
- [FastF1](https://github.com/theOehrly/Fast-F1)
