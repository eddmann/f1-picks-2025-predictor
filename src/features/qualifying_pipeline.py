"""
Qualifying feature pipeline for F1 predictions.

Extends BaseFeaturePipeline with qualifying-specific features:
- Q1/Q2/Q3 progression patterns
- Practice session features (FP1, FP2, FP3)
- Sector time analysis
- Tyre compound features

Temporal constraints:
- Can use: FP1, FP2, FP3, historical qualifying/race data
- Cannot use: Current qualifying results (that's the target)
"""

import logging
from pathlib import Path

import pandas as pd

from src.features.base_pipeline import BaseFeaturePipeline
from src.features.practice_features import PracticeFeatureExtractor
from src.features.qualifying_features import QualifyingFeatureExtractor

logger = logging.getLogger(__name__)


class QualifyingFeaturePipeline(BaseFeaturePipeline):
    """
    Feature pipeline for qualifying predictions.

    Uses all practice sessions (FP1, FP2, FP3) and historical data.
    Includes Q1/Q2/Q3 progression features for qualifying-specific patterns.
    """

    session_type = "Q"

    def __init__(self, data_dir: Path | str = "data/fastf1"):
        """
        Initialize the qualifying feature pipeline.

        Args:
            data_dir: Directory containing FastF1 parquet files
        """
        super().__init__(data_dir)

        # Initialize qualifying-specific extractors
        self.qualifying_extractor = QualifyingFeatureExtractor()
        self.practice_extractor = PracticeFeatureExtractor()

    def get_available_practice_sessions(self) -> list[str]:
        """All practice sessions are available before qualifying."""
        return ["FP1", "FP2", "FP3"]

    def get_available_current_weekend_sessions(self) -> list[str]:
        """No session results from current weekend (qualifying is the target)."""
        return []

    def _load_target_results(self, min_year: int) -> pd.DataFrame:
        """Load qualifying results as the prediction target."""
        return self.loader.load_qualifying_results(min_year=min_year)

    def _add_circuit_overtaking_features(
        self,
        df: pd.DataFrame,
        target_results: pd.DataFrame,
        min_year: int,
    ) -> pd.DataFrame:
        """
        Skip circuit overtaking features for qualifying.

        Overtaking features have 0% importance for qualifying prediction
        since qualifying is a pure time-trial format with no on-track passing.
        """
        logger.info("Skipping circuit overtaking features for qualifying (not applicable)")
        return df

    def _get_session_specific_features(
        self,
        df: pd.DataFrame,
        min_year: int,
        windows: list[int],
    ) -> pd.DataFrame:
        """
        Add qualifying-specific features.

        Includes:
        - Q1/Q2/Q3 progression features
        - Practice session features (FP1-FP3 lap times and rankings)
        - Sector time features
        - Tyre compound features
        - Sprint features (if sprint weekend)
        - Weather features
        """
        # Load session data
        quali_sessions = self.loader.load_qualifying_sessions(min_year=min_year)
        practice_sessions = self.loader.load_practice_sessions(min_year=min_year)
        qualifying_results = self._load_target_results(min_year=min_year)

        # Load sprint data (2021+)
        sprint_results = self.loader.load_sprint_results(min_year=max(min_year, 2021))
        sprint_quali_results = self.loader.load_sprint_qualifying_results(
            min_year=max(min_year, 2021)
        )

        # Sector features (historical sector time performance)
        logger.info("Extracting sector features...")
        sector_features = self.sector_extractor.extract_features(quali_sessions)
        if not sector_features.empty:
            df = self._merge_features(df, sector_features, "sector")

        # Qualifying progression features (Q1→Q2→Q3 patterns)
        logger.info("Extracting qualifying progression features...")
        quali_features = self.qualifying_extractor.extract_features(quali_sessions)
        if not quali_features.empty:
            df = self._merge_features(df, quali_features, "quali")

        # Practice features (FP1/FP2/FP3 correlation with qualifying)
        logger.info("Extracting practice features...")
        practice_features = self.practice_extractor.extract_features(
            practice_sessions, quali_sessions
        )
        if not practice_features.empty:
            df = self._merge_features(df, practice_features, "practice")

        # Tyre features (compound preferences)
        logger.info("Extracting tyre features...")
        tyre_features = self.tyre_extractor.extract_features(quali_sessions, practice_sessions)
        if not tyre_features.empty:
            df = self._merge_features(df, tyre_features, "tyre")

        # Sprint features (if sprint weekend)
        logger.info("Extracting sprint features...")
        sprint_features = self.sprint_extractor.extract_features(
            sprint_results, sprint_quali_results, qualifying_results
        )
        if not sprint_features.empty:
            df = self._merge_features(df, sprint_features, "sprint")

        # Weather features
        logger.info("Extracting weather features...")
        weather_features = self.weather_extractor.extract_features(
            quali_sessions, practice_sessions
        )
        if not weather_features.empty:
            df = self._merge_features(df, weather_features, "weather")

        return df

    def _get_feature_columns(self, windows: list[int]) -> list[str]:
        """Get all feature columns including qualifying-specific ones."""
        # Start with base features
        feature_cols = super()._get_feature_columns(windows)

        # Remove circuit overtaking features (not applicable for qualifying)
        overtaking_features = {
            "circuit_avg_positions_changed",
            "circuit_max_positions_changed",
            "circuit_positions_changed_std",
            "circuit_overtaking_rate",
            "circuit_difficulty",
            "grid_importance",
            "circuit_race_count",
            "low_overtaking_circuit",
            "high_overtaking_circuit",
            "driver_career_avg_overtakes",
            "driver_career_overtake_rate",
            "overtaking_specialist",
            "position_defender",
        }
        for w in windows:
            overtaking_features.update(
                [
                    f"driver_avg_overtakes_{w}",
                    f"driver_overtakes_std_{w}",
                    f"driver_gain_rate_{w}",
                    f"driver_max_overtakes_{w}",
                    f"overtake_potential_{w}",
                ]
            )
        feature_cols = [c for c in feature_cols if c not in overtaking_features]

        # Add sector features
        for w in windows:
            feature_cols.extend(
                [
                    f"rolling_best_s1_ms_{w}",
                    f"rolling_best_s2_ms_{w}",
                    f"rolling_best_s3_ms_{w}",
                    f"rolling_s1_rank_{w}",
                    f"rolling_s2_rank_{w}",
                    f"rolling_s3_rank_{w}",
                    f"rolling_s1_vs_field_{w}",
                    f"rolling_s2_vs_field_{w}",
                    f"rolling_s3_vs_field_{w}",
                    f"rolling_theoretical_gap_ms_{w}",
                    f"sector_balance_{w}",
                ]
            )
        feature_cols.append("strongest_sector")

        # Add qualifying progression features
        for w in windows:
            feature_cols.extend(
                [
                    f"q2_rate_{w}",
                    f"q3_rate_{w}",
                    f"top3_rate_{w}",
                    f"top10_rate_{w}",
                    f"pole_rate_{w}",
                    f"front_row_rate_{w}",
                    f"avg_position_{w}",
                    f"position_std_{w}",
                    f"avg_q1_to_q2_improvement_{w}",
                    f"avg_q2_to_q3_improvement_{w}",
                    f"avg_q1_margin_{w}",
                    f"avg_q2_margin_{w}",
                ]
            )
        feature_cols.append("final_run_specialist")

        # Add practice features (current weekend FP1-FP3)
        feature_cols.extend(
            [
                "current_fp1_best_ms",
                "current_fp2_best_ms",
                "current_fp3_best_ms",
                "current_practice_best_ms",
                "current_total_practice_laps",
                "current_fp1_to_fp3_improvement_pct",
                # FP ranking features (relative to field - highly predictive)
                "current_fp3_rank",
                "current_fp3_gap_ms",
                "current_fp3_gap_pct",
                "current_fp2_rank",
                "current_fp2_gap_ms",
                "current_fp2_gap_pct",
                "current_practice_rank",
                "current_practice_gap_ms",
                "current_practice_gap_pct",
            ]
        )
        for w in windows:
            feature_cols.extend(
                [
                    f"avg_practice_to_quali_gap_{w}",
                    f"avg_fp3_to_quali_gap_{w}",
                    f"avg_fp_improvement_{w}",
                    f"avg_long_run_pace_{w}",
                ]
            )

        # Add tyre features
        feature_cols.extend(
            [
                "current_used_soft",
                "current_used_medium",
                "current_compounds_used",
            ]
        )
        for w in windows:
            feature_cols.extend(
                [
                    f"soft_usage_rate_{w}",
                    f"avg_soft_advantage_{w}",
                    f"avg_fresh_advantage_{w}",
                    f"avg_best_lap_tyre_life_{w}",
                ]
            )

        # Add sprint features
        feature_cols.append("is_sprint_weekend")
        for w in windows:
            feature_cols.extend(
                [
                    f"sprint_avg_position_{w}",
                    f"sprint_consistency_{w}",
                    f"sprint_positions_gained_{w}",
                    f"sq_avg_position_{w}",
                    f"sq_consistency_{w}",
                    f"sq_to_q_improvement_{w}",
                ]
            )

        # Add weather features
        feature_cols.extend(
            [
                "current_track_temp",
                "current_air_temp",
                "current_humidity",
                "is_wet_session",
                "track_temp_vs_season",
                "fp_to_q_temp_delta",
            ]
        )
        for w in windows:
            feature_cols.extend(
                [
                    f"wet_session_rate_{w}",
                    f"hot_track_avg_pos_{w}",
                    f"cold_track_avg_pos_{w}",
                ]
            )

        return feature_cols


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    pipeline = QualifyingFeaturePipeline()
    X, y, meta = pipeline.build_features(min_year=2024)

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Sample features: {list(X.columns)[:20]}")
    print("\nTarget distribution:")
    print(y.value_counts().head(10))
