"""
Smart feature imputation strategies for F1 prediction features.

Provides semantically-aware imputation that respects the meaning of different
feature types rather than using a blanket fillna(0) which can be misleading.

Problem with fillna(0):
- Position features: 0 implies P1 (first place), but NaN means "no data"
- Time features: 0ms implies perfect lap time, but NaN means "no data"
- Rate features: 0 is correct (no history = 0% rate)
- Count features: 0 is correct (no appearances = 0)

This module provides proper imputation for each feature type.
"""

import logging
from dataclasses import dataclass
from enum import Enum

import pandas as pd

logger = logging.getLogger(__name__)


class ImputationStrategy(Enum):
    """Imputation strategies for different feature types."""

    CONSTANT = "constant"  # Use a specific constant value
    MEDIAN = "median"  # Use column median from training data
    FIELD_WORST_MARGIN = "field_worst_margin"  # Use worst value + margin (for times)
    ZERO = "zero"  # Use 0 (semantically correct for rates/counts)


@dataclass
class FeatureImputationConfig:
    """Configuration for feature imputation."""

    strategy: ImputationStrategy
    constant_value: float | None = None
    margin_pct: float = 0.05  # For FIELD_WORST_MARGIN strategy


# Feature pattern to imputation strategy mapping
FEATURE_IMPUTATION_RULES: dict[str, FeatureImputationConfig] = {
    # Position features - use P10 as neutral mid-grid
    "rolling_avg_pos_": FeatureImputationConfig(ImputationStrategy.CONSTANT, 10.0),
    "avg_position_": FeatureImputationConfig(ImputationStrategy.CONSTANT, 10.0),
    "team_rolling_avg_pos": FeatureImputationConfig(ImputationStrategy.CONSTANT, 10.0),
    "team_rolling_best_pos": FeatureImputationConfig(ImputationStrategy.CONSTANT, 5.0),
    "driver_circuit_avg": FeatureImputationConfig(ImputationStrategy.CONSTANT, 10.0),
    "team_circuit_avg": FeatureImputationConfig(ImputationStrategy.CONSTANT, 10.0),
    "sprint_avg_position_": FeatureImputationConfig(ImputationStrategy.CONSTANT, 10.0),
    "sq_avg_position_": FeatureImputationConfig(ImputationStrategy.CONSTANT, 10.0),
    # Time features (ms) - use field worst + margin
    "rolling_best_s1_ms_": FeatureImputationConfig(ImputationStrategy.FIELD_WORST_MARGIN),
    "rolling_best_s2_ms_": FeatureImputationConfig(ImputationStrategy.FIELD_WORST_MARGIN),
    "rolling_best_s3_ms_": FeatureImputationConfig(ImputationStrategy.FIELD_WORST_MARGIN),
    "current_fp": FeatureImputationConfig(ImputationStrategy.FIELD_WORST_MARGIN),
    "current_practice_best_ms": FeatureImputationConfig(ImputationStrategy.FIELD_WORST_MARGIN),
    "avg_long_run_pace_": FeatureImputationConfig(ImputationStrategy.FIELD_WORST_MARGIN),
    "rolling_theoretical_best_ms_": FeatureImputationConfig(ImputationStrategy.FIELD_WORST_MARGIN),
    # Rate features - 0 is correct (no history = 0% rate)
    "rolling_top3_rate_": FeatureImputationConfig(ImputationStrategy.ZERO),
    "rolling_pole_rate_": FeatureImputationConfig(ImputationStrategy.ZERO),
    "rolling_front_row_rate_": FeatureImputationConfig(ImputationStrategy.ZERO),
    "q2_rate_": FeatureImputationConfig(ImputationStrategy.ZERO),
    "q3_rate_": FeatureImputationConfig(ImputationStrategy.ZERO),
    "top3_rate_": FeatureImputationConfig(ImputationStrategy.ZERO),
    "top10_rate_": FeatureImputationConfig(ImputationStrategy.ZERO),
    "pole_rate_": FeatureImputationConfig(ImputationStrategy.ZERO),
    "front_row_rate_": FeatureImputationConfig(ImputationStrategy.ZERO),
    "soft_usage_rate_": FeatureImputationConfig(ImputationStrategy.ZERO),
    # Count features - 0 is correct
    "season_race_count": FeatureImputationConfig(ImputationStrategy.ZERO),
    "career_race_count": FeatureImputationConfig(ImputationStrategy.ZERO),
    "circuit_appearances": FeatureImputationConfig(ImputationStrategy.ZERO),
    "total_practice_laps": FeatureImputationConfig(ImputationStrategy.ZERO),
    # Consistency (std) - use median as neutral
    "rolling_pos_std_": FeatureImputationConfig(ImputationStrategy.MEDIAN),
    "position_std_": FeatureImputationConfig(ImputationStrategy.MEDIAN),
    "sprint_consistency_": FeatureImputationConfig(ImputationStrategy.MEDIAN),
    "sq_consistency_": FeatureImputationConfig(ImputationStrategy.MEDIAN),
    "sector_balance_": FeatureImputationConfig(ImputationStrategy.MEDIAN),
    # Rank features - use P10 as neutral
    "rolling_s1_rank_": FeatureImputationConfig(ImputationStrategy.CONSTANT, 10.0),
    "rolling_s2_rank_": FeatureImputationConfig(ImputationStrategy.CONSTANT, 10.0),
    "rolling_s3_rank_": FeatureImputationConfig(ImputationStrategy.CONSTANT, 10.0),
    # Gap/Delta features - 0 is neutral (neither ahead nor behind)
    "vs_field": FeatureImputationConfig(ImputationStrategy.ZERO),
    "avg_q1_margin_": FeatureImputationConfig(ImputationStrategy.ZERO),
    "avg_q2_margin_": FeatureImputationConfig(ImputationStrategy.ZERO),
    "sq_to_q_improvement_": FeatureImputationConfig(ImputationStrategy.ZERO),
    "theoretical_gap_ms": FeatureImputationConfig(ImputationStrategy.MEDIAN),
    "practice_to_quali_gap": FeatureImputationConfig(ImputationStrategy.MEDIAN),
    "fp3_to_quali_gap": FeatureImputationConfig(ImputationStrategy.MEDIAN),
    # Binary flags - 0 (false/no)
    "is_sprint_weekend": FeatureImputationConfig(ImputationStrategy.ZERO),
    "current_used_soft": FeatureImputationConfig(ImputationStrategy.ZERO),
    "current_used_medium": FeatureImputationConfig(ImputationStrategy.ZERO),
    "is_wet_session": FeatureImputationConfig(ImputationStrategy.ZERO),
    # Improvement percentages - 0 is neutral
    "improvement_pct": FeatureImputationConfig(ImputationStrategy.ZERO),
    "avg_fp_improvement_": FeatureImputationConfig(ImputationStrategy.ZERO),
    "avg_q1_to_q2_improvement_": FeatureImputationConfig(ImputationStrategy.ZERO),
    "avg_q2_to_q3_improvement_": FeatureImputationConfig(ImputationStrategy.ZERO),
    "avg_soft_advantage_": FeatureImputationConfig(ImputationStrategy.ZERO),
    "avg_fresh_advantage_": FeatureImputationConfig(ImputationStrategy.ZERO),
    # Momentum - 0 is neutral (no trend)
    "momentum": FeatureImputationConfig(ImputationStrategy.ZERO),
    "position_trend_": FeatureImputationConfig(ImputationStrategy.ZERO),
    "form_acceleration_": FeatureImputationConfig(ImputationStrategy.ZERO),
    # Specialist scores - 0 (no pattern)
    "final_run_specialist": FeatureImputationConfig(ImputationStrategy.ZERO),
    # Sprint positions gained - 0 (no gain/loss)
    "sprint_positions_gained_": FeatureImputationConfig(ImputationStrategy.ZERO),
    # Tyre life - use median
    "avg_best_lap_tyre_life_": FeatureImputationConfig(ImputationStrategy.MEDIAN),
    # Weather features
    "current_track_temp": FeatureImputationConfig(ImputationStrategy.MEDIAN),
    "current_air_temp": FeatureImputationConfig(ImputationStrategy.MEDIAN),
    "current_humidity": FeatureImputationConfig(ImputationStrategy.MEDIAN),
    "track_temp_normalized": FeatureImputationConfig(ImputationStrategy.ZERO),
    # Relative features
    "position_vs_field_avg": FeatureImputationConfig(ImputationStrategy.ZERO),
    "position_percentile_": FeatureImputationConfig(ImputationStrategy.CONSTANT, 0.5),
    "teammate_gap_ms_": FeatureImputationConfig(ImputationStrategy.ZERO),
    "beats_teammate_rate_": FeatureImputationConfig(ImputationStrategy.CONSTANT, 0.5),
    "teammate_position_delta_": FeatureImputationConfig(ImputationStrategy.ZERO),
    # EWM features
    "ewm_position_": FeatureImputationConfig(ImputationStrategy.CONSTANT, 10.0),
    "ewm_top3_rate_": FeatureImputationConfig(ImputationStrategy.ZERO),
}


class FeatureImputer:
    """Smart imputer for F1 prediction features."""

    def __init__(
        self,
        rules: dict[str, FeatureImputationConfig] | None = None,
        default_strategy: ImputationStrategy = ImputationStrategy.ZERO,
    ):
        """
        Initialize the imputer.

        Args:
            rules: Custom imputation rules. Defaults to FEATURE_IMPUTATION_RULES.
            default_strategy: Strategy for features not matching any rule.
        """
        self.rules = rules or FEATURE_IMPUTATION_RULES
        self.default_strategy = default_strategy
        self._fitted_values: dict[str, float] = {}
        self._is_fitted = False

    def fit(self, X: pd.DataFrame) -> "FeatureImputer":
        """
        Fit the imputer by computing necessary statistics.

        Args:
            X: Feature DataFrame to fit on

        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting imputer on {len(X.columns)} features")

        for col in X.columns:
            config = self._get_config(col)

            if config.strategy == ImputationStrategy.MEDIAN:
                median_val = X[col].median()
                self._fitted_values[col] = median_val if pd.notna(median_val) else 0.0
            elif config.strategy == ImputationStrategy.FIELD_WORST_MARGIN:
                # Use max (worst time) + margin
                max_val = X[col].max()
                if pd.notna(max_val) and max_val > 0:
                    self._fitted_values[col] = max_val * (1 + config.margin_pct)
                else:
                    # Fallback to median if no valid max
                    median_val = X[col].median()
                    self._fitted_values[col] = median_val if pd.notna(median_val) else 0.0

        self._is_fitted = True
        logger.info(f"Imputer fitted, stored {len(self._fitted_values)} computed values")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply imputation to features.

        Args:
            X: Feature DataFrame to transform

        Returns:
            DataFrame with NaN values imputed
        """
        if not self._is_fitted:
            raise ValueError("Imputer must be fitted before transform")

        X_imputed = X.copy()
        imputed_count = 0

        for col in X_imputed.columns:
            nan_count = X_imputed[col].isna().sum()
            if nan_count > 0:
                fill_value = self._get_fill_value(col)
                X_imputed[col] = X_imputed[col].fillna(fill_value)
                imputed_count += nan_count

        logger.debug(f"Imputed {imputed_count} NaN values across {len(X.columns)} features")
        return X_imputed

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def _get_config(self, col_name: str) -> FeatureImputationConfig:
        """Get imputation config for a column."""
        # Check for exact match first
        if col_name in self.rules:
            return self.rules[col_name]

        # Check for prefix/substring matches
        for pattern, config in self.rules.items():
            if col_name.startswith(pattern) or pattern in col_name:
                return config

        # Default fallback
        return FeatureImputationConfig(self.default_strategy)

    def _get_fill_value(self, col_name: str) -> float:
        """Get the fill value for a column."""
        config = self._get_config(col_name)

        if config.strategy == ImputationStrategy.ZERO:
            return 0.0
        elif config.strategy == ImputationStrategy.CONSTANT:
            return config.constant_value if config.constant_value is not None else 0.0
        elif config.strategy in (ImputationStrategy.MEDIAN, ImputationStrategy.FIELD_WORST_MARGIN):
            return self._fitted_values.get(col_name, 0.0)
        else:
            return 0.0

    def get_imputation_summary(self) -> pd.DataFrame:
        """
        Get a summary of imputation rules applied.

        Returns:
            DataFrame with column, strategy, and fill value
        """
        if not self._is_fitted:
            raise ValueError("Imputer must be fitted first")

        summary = []
        for col, value in self._fitted_values.items():
            config = self._get_config(col)
            summary.append(
                {
                    "column": col,
                    "strategy": config.strategy.value,
                    "fill_value": value,
                }
            )

        return pd.DataFrame(summary)
