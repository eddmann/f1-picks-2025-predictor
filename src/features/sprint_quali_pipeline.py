"""
Sprint Qualifying feature pipeline for F1 predictions.

Extends BaseFeaturePipeline with sprint qualifying-specific features.

CRITICAL TEMPORAL CONSTRAINTS:
Sprint qualifying happens BEFORE FP2/FP3 and main qualifying.
In 2024 format: FP1 -> SQ -> Sprint Race -> FP2 -> FP3 -> Qualifying -> Race

Available data for SQ prediction:
- FP1 only (not FP2/FP3 - they happen after SQ)
- Historical SQ/Q/R data
- NO current weekend results

This is the most constrained pipeline with limited feature availability.
"""

import logging
from pathlib import Path

import pandas as pd

from src.features.base_pipeline import BaseFeaturePipeline

logger = logging.getLogger(__name__)


class SprintQualiFeaturePipeline(BaseFeaturePipeline):
    """
    Feature pipeline for sprint qualifying predictions.

    Uses only FP1 from current weekend (FP2/FP3 happen after sprint quali).
    Relies heavily on historical performance features due to limited current data.

    Sprint qualifying has limited data (6 weekends/year since 2021), so
    models should use stronger regularization.
    """

    session_type = "SQ"

    def __init__(self, data_dir: Path | str = "data/fastf1"):
        """
        Initialize the sprint qualifying feature pipeline.

        Args:
            data_dir: Directory containing FastF1 parquet files
        """
        super().__init__(data_dir)

    def get_available_practice_sessions(self) -> list[str]:
        """Only FP1 is available before sprint qualifying."""
        return ["FP1"]

    def get_available_current_weekend_sessions(self) -> list[str]:
        """No session results available before sprint qualifying."""
        return []

    def _load_target_results(self, min_year: int) -> pd.DataFrame:
        """Load sprint qualifying results as the prediction target."""
        # Sprint format started in 2021
        effective_min_year = max(min_year, 2021)
        return self.loader.load_sprint_qualifying_results(min_year=effective_min_year)

    def _get_session_specific_features(
        self,
        df: pd.DataFrame,
        min_year: int,
        windows: list[int],
    ) -> pd.DataFrame:
        """
        Add sprint qualifying-specific features.

        IMPORTANT: Only FP1 data from current weekend is available.
        FP2, FP3, and main qualifying happen AFTER sprint qualifying.

        Includes:
        - FP1-only practice features
        - Historical SQ performance
        - Historical main qualifying performance (correlated with SQ)
        - Historical sprint race performance
        - Weather features (from FP1 only)
        """
        effective_min_year = max(min_year, 2021)

        # Load session data
        practice_sessions = self.loader.load_practice_sessions(min_year=min_year)
        quali_sessions = self.loader.load_qualifying_sessions(min_year=min_year)
        qualifying_results = self.loader.load_qualifying_results(min_year=min_year)

        # Load sprint data
        sprint_results = self.loader.load_sprint_results(min_year=effective_min_year)
        sprint_quali_results = self.loader.load_sprint_qualifying_results(
            min_year=effective_min_year
        )

        # FP1-only practice features (CRITICAL: no FP2/FP3!)
        logger.info("Extracting FP1-only practice features...")
        df = self._add_fp1_only_features(df, practice_sessions)

        # Historical sector features from main qualifying
        logger.info("Extracting historical sector features...")
        sector_features = self.sector_extractor.extract_features(quali_sessions)
        if not sector_features.empty:
            df = self._merge_features(df, sector_features, "sector")

        # Historical sprint features
        logger.info("Extracting historical sprint features...")
        sprint_features = self.sprint_extractor.extract_features(
            sprint_results, sprint_quali_results, qualifying_results
        )
        if not sprint_features.empty:
            df = self._merge_features(df, sprint_features, "sprint")

        # Add historical main qualifying correlation features
        df = self._add_historical_quali_correlation(df, qualifying_results)

        # Weather features (from FP1 only)
        logger.info("Extracting weather features...")
        fp1_sessions = practice_sessions[practice_sessions["session_type"] == "FP1"]
        if not fp1_sessions.empty:
            weather_features = self.weather_extractor.extract_features(fp1_sessions, fp1_sessions)
            if not weather_features.empty:
                df = self._merge_features(df, weather_features, "weather")

        return df

    def _add_fp1_only_features(
        self,
        df: pd.DataFrame,
        practice_sessions: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add features from FP1 only.

        FP2 and FP3 happen AFTER sprint qualifying, so we cannot use them.
        """
        if practice_sessions.empty:
            return df

        # Filter to FP1 only
        fp1_sessions = practice_sessions[practice_sessions["session_type"] == "FP1"].copy()

        if fp1_sessions.empty:
            return df

        # Get best FP1 lap time per driver per weekend
        fp1_best = (
            fp1_sessions.groupby(["year", "round", "driver_code"])
            .agg(
                {
                    "lap_time_ms": "min",
                    "lap_number": "count",
                }
            )
            .reset_index()
        )
        fp1_best = fp1_best.rename(
            columns={
                "lap_time_ms": "current_fp1_best_ms",
                "lap_number": "current_fp1_laps",
            }
        )

        # Calculate FP1 rankings per weekend
        fp1_best["current_fp1_rank"] = fp1_best.groupby(["year", "round"])[
            "current_fp1_best_ms"
        ].rank()

        # Calculate gap to FP1 leader
        fp1_best["fp1_leader_time"] = fp1_best.groupby(["year", "round"])[
            "current_fp1_best_ms"
        ].transform("min")
        fp1_best["current_fp1_gap_ms"] = (
            fp1_best["current_fp1_best_ms"] - fp1_best["fp1_leader_time"]
        )
        fp1_best["current_fp1_gap_pct"] = (
            fp1_best["current_fp1_gap_ms"] / fp1_best["fp1_leader_time"] * 100
        )

        # Drop helper column
        fp1_best = fp1_best.drop(columns=["fp1_leader_time"])

        # Merge with main dataframe
        df = df.merge(
            fp1_best,
            on=["year", "round", "driver_code"],
            how="left",
        )

        return df

    def _add_historical_quali_correlation(
        self,
        df: pd.DataFrame,
        qualifying_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add historical qualifying features that correlate with SQ performance.

        Main qualifying and sprint qualifying performance are correlated,
        so historical main quali performance helps predict SQ.
        """
        if qualifying_results.empty:
            return df

        # Calculate rolling qualifying position
        quali = qualifying_results.copy()
        quali["position"] = pd.to_numeric(quali["position"], errors="coerce")
        quali = quali.sort_values(["driver_code", "year", "round"])

        for window in [3, 5]:
            quali[f"historical_quali_avg_{window}"] = quali.groupby("driver_code")[
                "position"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

        # Merge with SQ data (using year/round/driver)
        quali_features = quali[
            ["year", "round", "driver_code"]
            + [c for c in quali.columns if c.startswith("historical_quali")]
        ].drop_duplicates(subset=["year", "round", "driver_code"])

        df = df.merge(
            quali_features,
            on=["year", "round", "driver_code"],
            how="left",
        )

        return df

    def _get_feature_columns(self, windows: list[int]) -> list[str]:
        """Get all feature columns including sprint quali-specific ones."""
        # Start with base features
        feature_cols = super()._get_feature_columns(windows)

        # FP1-only features (CRITICAL: no FP2/FP3 features!)
        feature_cols.extend(
            [
                "current_fp1_best_ms",
                "current_fp1_laps",
                "current_fp1_rank",
                "current_fp1_gap_ms",
                "current_fp1_gap_pct",
            ]
        )

        # Historical qualifying correlation features
        for w in [3, 5]:
            feature_cols.append(f"historical_quali_avg_{w}")

        # Sector features (from historical qualifying)
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

        # Sprint features (historical)
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

        # Weather features (from FP1 only)
        feature_cols.extend(
            [
                "current_track_temp",
                "current_air_temp",
                "current_humidity",
                "is_wet_session",
                "track_temp_vs_season",
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

    pipeline = SprintQualiFeaturePipeline()
    X, y, meta = pipeline.build_features(min_year=2021)

    print(f"\nFeature matrix shape: {X.shape}")
    print("\nFP1-only features:")
    fp1_cols = [c for c in X.columns if "fp1" in c.lower()]
    print(fp1_cols)
    print("\nNOTE: No FP2/FP3 features - they happen after sprint qualifying")
    print("\nTarget distribution (SQ positions):")
    print(y.value_counts().head(10))
