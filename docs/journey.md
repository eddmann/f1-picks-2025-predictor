# The Journey

This document chronicles how this F1 prediction system evolved - the experiments, the failures, and the insights. If you're building something similar or want to understand why certain decisions were made, this is for you.

## Phase 1: Foundation

**Goal:** Build a basic F1 data pipeline and get initial predictions working.

### Data Evolution

Started with the [Kaggle F1 dataset](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) - comprehensive historical data but static (only updated annually). Migrated to [FastF1](https://github.com/theOehrly/Fast-F1) for live data access, which enabled:

- Real-time practice session times
- Sector-level timing data
- Weather conditions
- Tyre compound information

### Model Evolution

Initial approach: **Classification** - predict "will this driver finish top-3?" for each driver.

**Problem:** Classification doesn't capture relative ordering. A classifier might say all top-10 drivers have 60% top-3 probability, but we need to rank them.

**Solution:** Switched to **LightGBM LambdaRank** - a learning-to-rank model that directly optimizes driver ordering within each session.

```
Classification: "Driver A has 70% top-3 probability"
Ranking: "Driver A should be ranked above Driver B"
```

The ranker learns to compare drivers against each other, which is exactly what we need.

---

## Phase 2: The Baseline Reality Check

**Goal:** Understand what we're actually trying to beat.

This phase was humbling. We implemented trivial baselines and discovered that simple heuristics are surprisingly effective:

| Baseline               | Score     | What It Does                            |
| ---------------------- | --------- | --------------------------------------- |
| **Championship Top-3** | ~2.37 pts | Always pick current standings leaders   |
| **FP3 Fastest**        | 2.64 pts  | Pick the three fastest drivers in FP3   |
| **Grid Order**         | 3.20 pts  | For races, just use qualifying P1-P2-P3 |

**Key Insight:** If our ML model can't beat "just pick whoever was fastest in practice," it provides zero value. This framing guided all subsequent development.

---

## Phase 3: Feature Engineering - What Worked

### ELO Ratings (+2-4% improvement)

Inspired by chess ELO ratings and [research from the F1 prediction community](https://matthewperron.github.io/f1-elo/).

**The Problem:** In F1, you can't directly compare drivers across teams because car performance varies drastically. Max Verstappen in a dominant Red Bull vs. a backmarker driver isn't a fair comparison.

**The Solution:** Only compare teammates (same car) and track these relative performance ratings over time.

**Features implemented (10 total):**

- `driver_elo` - Skill rating from teammate comparisons
- `constructor_elo` - Team/car performance from field comparisons
- `ga_driver_elo` - Grid-adjusted ELO (accounts for starting position)
- `combined_elo` - 30% driver + 70% constructor (car matters more in F1)
- Percentile and delta variants

**Results:**

- Qualifying: 2.56 → 2.62 pts (+2.3%)
- Race: 2.78 → 2.88 pts (+3.6%)

ELO features ranked 4th and 6th in qualifying model importance.

### Reliability Features (+2% for race)

Inspired by [F1-Predictor](https://github.com/JaideepGuntupalli/f1-predictor). The insight: reliable drivers/teams are more likely to finish races.

**Features implemented (24 total):**

- `driver_dnf_rate_{window}` - Rolling DNF rates
- `driver_mechanical_dnf_rate` - Engine, gearbox, suspension failures
- `driver_incident_dnf_rate` - Crashes, collisions (driver error indicator)
- `team_reliability` - Team's mechanical failure rate
- `races_since_dnf` - Consecutive finishes

**Key Finding:** DNF features help race prediction (50+ laps where reliability matters) but have zero impact on qualifying (single lap performance).

### First Lap Features (+6.5% for qualifying!)

This was the breakthrough that finally let qualifying beat the FP3 baseline.

**The Hypothesis:** Drivers who consistently gain positions on lap 1 might have skills that correlate with qualifying performance - aggression, racecraft, consistency.

**The Discovery:** During implementation, we found that FastF1's lap-level `position` column contains **final race position repeated across all laps**, not actual lap-by-lap positions. This was a data quality issue that took time to diagnose.

**The Workaround:**

- For qualifying predictions, first lap features provided genuinely new signal
- For race predictions, they were perfectly correlated with existing position-gained features (correlation = 1.0)
- Solution: Only apply first lap features to Qualifying and Sprint pipelines

**Results:**

- Qualifying: 2.62 → 2.79 pts (+6.5%)
- **Qualifying now beats FP3 baseline** (2.79 vs 2.64, +5.7%)

### Circuit-Type Features (+4.8% for race)

Different drivers excel on different circuit types:

- **Street circuits:** Monaco, Singapore, Baku
- **High-speed:** Monza, Spa, Silverstone
- **Technical:** Hungary, Barcelona

**Features implemented (13 total):**

- `circuit_type_avg_position` - Historical performance at this circuit type
- `circuit_type_affinity` - How much better/worse vs overall performance
- `circuit_teammate_delta` - Gap to teammate at this circuit type

**Results:** Race model improved 2.93 → 3.07 pts (+4.8%). Modest but meaningful.

---

## Phase 4: The Race Prediction Wall

This phase taught us the limits of ML for race prediction.

### The Two-Stage Experiment

We tried separating the "grid effect" from "racing factors":

**Stage 1:** Learn historical grid-to-finish conversion rates

- P1 on grid → P1 finish 70% of the time
- P5 on grid → P1 finish 5% of the time

**Stage 2:** ML model predicts who will over/underperform their grid position

### The Devastating Finding

Stage 1 alone (just using grid position) achieved **3.17 pts**. Adding ANY amount of ML predictions made it worse:

| Alpha (ML weight) | Score    | Description               |
| ----------------- | -------- | ------------------------- |
| 0.0               | **3.17** | Grid expected finish only |
| 0.1               | 3.16     | 10% ML weight             |
| 0.5               | 3.15     | 50/50 blend               |
| 1.0               | 2.95     | Equal weight              |
| 2.0               | 2.36     | ML dominant               |

### Why Race Prediction Has a Ceiling

Analysis revealed that race outcomes have ~40% unpredictable variance:

1. **Safety car timing** - Random events that bunch up the field
2. **Weather changes** - Conditions during quali ≠ conditions during race
3. **Mechanical failures** - Random engine/gearbox failures
4. **First lap chaos** - Multi-car incidents are unpredictable
5. **Strategy game theory** - Pit stop timing is a complex multi-agent game

**30% of podiums come from drivers starting outside grid top-3.** Of these "upsets," only 27% were from drivers with historically high position-gain rates. The remaining 73% were driven by unpredictable race events.

---

## Phase 5: Hyperparameter Tuning (+7.6%)

Used Optuna Bayesian optimization with temporal cross-validation.

### Qualifying Model Changes

- `n_estimators`: 100 → 259
- `num_leaves`: 31 → 15 (simpler trees, less overfitting)
- `learning_rate`: 0.1 → 0.082

### Race Model Changes

- `n_estimators`: 100 → 192
- `num_leaves`: 31 → 20
- Added `reg_lambda`: 0.326 (L2 regularization)

**Results:** Qualifying CV score improved 2.75 → 2.96 pts (+7.6%)

**Lesson:** Feature engineering provides bigger gains than hyperparameter tuning, but tuning helps polish a mature feature set.

---

## Phase 6: Expanding the Feature Set

**Goal:** Extract more signal from conditions that create performance variance.

### Wet Weather Skill Features (+31 features)

Some drivers consistently outperform in wet conditions (Hamilton, Verstappen) while others struggle. Wet races create the biggest upsets, so capturing this skill differential seemed valuable.

**Features implemented:**

- `wet_avg_position_{w}` - Rolling average position in wet sessions
- `wet_vs_dry_delta` - Performance gap between wet and dry conditions
- `wet_session_count` - Experience in wet conditions
- `wet_vs_field` - Relative wet performance vs the field
- `intermediate_specialist` - Performance specifically on intermediate tyres

**Key insight:** Driver skill matters more in wet conditions because car advantage is reduced. The best wet weather drivers can gain 3-5 positions relative to their dry performance.

### Track Evolution Features (+18 features)

Track grip improves throughout a session as more rubber is laid down. This creates advantages for drivers who run later in sessions - Q3 track is significantly grippier than Q1.

**Features implemented:**

- `track_evolution_gain_{w}` - How much driver improves from early to late runs
- `green_track_specialist` - Performance on fresh/low-grip surfaces
- `rubbered_track_specialist` - Performance on evolved high-grip tracks
- `adaptation_rate` - Speed of adapting to changing grip levels

**Key insight:** Lap times can improve 1-2 seconds from FP1 to Q3 due to rubber buildup. Some drivers excel at extracting performance early (useful for qualifying shootouts), while others need track evolution.

### Circuit Overtaking Features (+28 features)

Some circuits are notoriously difficult to overtake on (Monaco, Hungary), while others promote position changes (Monza, Bahrain). This affects how much grid position matters.

**Features implemented:**

- `circuit_overtake_rate` - Historical overtaking frequency at this circuit
- `driver_overtake_rate_{w}` - Driver's historical positions gained per race
- `driver_vs_circuit_overtake` - How driver's overtaking compares to circuit average
- `grid_importance_factor` - How much qualifying matters at this circuit

**Key insight:** At Monaco, starting P1 almost guarantees a win. At Monza, a fast car from P10 can realistically reach the podium. The features help the model weight grid position appropriately per circuit.

### Results

| Model        | Before   | After        | Change      |
| ------------ | -------- | ------------ | ----------- |
| Qualifying   | 2.96 pts | **3.18 pts** | **+7.4%**   |
| Race         | 2.80 pts | 2.80 pts     | No change   |

The new features primarily benefited qualifying prediction. Race prediction remained constrained by its fundamental ceiling - these features don't help predict safety cars or mechanical failures.

---

## What Didn't Work

| Approach                   | Why It Failed                                                                |
| -------------------------- | ---------------------------------------------------------------------------- |
| **Sector times for race**  | Single-lap metrics don't predict 50+ lap races (0% importance in race model) |
| **Tyre compound features** | Qualifying tyre usage has no relationship to race outcome                    |
| **Weather features**       | Weather at qualifying ≠ weather at race                                      |
| **Two-stage race model**   | ML predictions actively hurt performance vs grid-only                        |
| **Classification models**  | Don't capture relative ordering between drivers                              |

---

## Key Learnings

1. **Qualifying is learnable** - Combining FP3 pace with ELO ratings and historical form can modestly beat practice-only baselines.

2. **Race is fundamentally limited** - Grid position explains ~60% of outcome. The remaining 40% involves genuinely unpredictable factors.

3. **Data quality matters** - The FastF1 lap position bug cost significant development time. Always validate your data.

4. **Feature pruning helps** - Removing 40-80% of zero-importance features improved scores 2-7%. Less noise = better generalization.

5. **Baselines are essential** - Without proper baselines, we'd have thought 2.5 pts was good performance. It's not.

6. **Temporal integrity is critical** - Using `.shift(1).rolling()` patterns prevents data leakage. Without this, cross-validation scores are meaningless.

---

## References

- [F1 ELO by Matthew Perron](https://matthewperron.github.io/f1-elo/) - Teammate comparison methodology
- [F1 Rating System](https://github.com/mwtmurphy/f1-elo) - Separate driver/constructor ratings
- [F1-Predictor](https://github.com/JaideepGuntupalli/f1-predictor) - Feature engineering ideas
- [FastF1](https://github.com/theOehrly/Fast-F1) - F1 timing data access
