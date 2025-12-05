"""
Sprint Race feature pipeline for F1 predictions.

Extends BaseFeaturePipeline with sprint race-specific features.

CRITICAL TEMPORAL CONSTRAINTS:
Sprint race happens AFTER sprint qualifying but BEFORE FP2/FP3 and main qualifying.
In 2024 format: FP1 -> SQ -> Sprint Race -> FP2 -> FP3 -> Qualifying -> Race

Available data for Sprint Race prediction:
- FP1 (happened before SQ)
- Sprint Qualifying results (grid for sprint race)
- Historical data
- NO FP2/FP3 (happen after sprint race)
- NO main qualifying (happens after sprint race)
"""

import logging
from pathlib import Path

import pandas as pd

from src.features.base_pipeline import BaseFeaturePipeline
from src.features.grid_features import GridPositionFeatureExtractor

logger = logging.getLogger(__name__)


class SprintRaceFeaturePipeline(BaseFeaturePipeline):
    """
    Feature pipeline for sprint race predictions.

    Uses FP1 and sprint qualifying results from current weekend.
    Sprint qualifying position is the grid position for the sprint race.

    Sprint races are shorter (~20 laps) than main races (~50-70 laps),
    so position changes are less common.
    """

    session_type = "S"

    def __init__(self, data_dir: Path | str = "data/fastf1"):
        """
        Initialize the sprint race feature pipeline.

        Args:
            data_dir: Directory containing FastF1 parquet files
        """
        super().__init__(data_dir)

        # Initialize sprint-specific extractors
        self.grid_extractor = GridPositionFeatureExtractor()

    def get_available_practice_sessions(self) -> list[str]:
        """Only FP1 is available before sprint race."""
        return ["FP1"]

    def get_available_current_weekend_sessions(self) -> list[str]:
        """Sprint qualifying results are available (sets sprint race grid)."""
        return ["SQ"]

    def _load_target_results(self, min_year: int) -> pd.DataFrame:
        """Load sprint race results as the prediction target."""
        # Sprint format started in 2021
        effective_min_year = max(min_year, 2021)
        return self.loader.load_sprint_results(min_year=effective_min_year)

    def _get_session_specific_features(
        self,
        df: pd.DataFrame,
        min_year: int,
        windows: list[int],
    ) -> pd.DataFrame:
        """
        Add sprint race-specific features.

        IMPORTANT: Only FP1 and SQ data from current weekend is available.
        FP2, FP3, and main qualifying happen AFTER sprint race.

        Includes:
        - Current sprint qualifying grid position (CRUCIAL - sets sprint grid)
        - FP1-only practice features
        - Historical sprint race performance
        - Historical grid-to-finish features from sprints
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

        # CRITICAL: Add current SQ grid position (sprint race grid)
        logger.info("Adding sprint qualifying grid position...")
        df = self._add_sprint_grid_features(df, sprint_quali_results)

        # Historical sprint grid-to-finish performance
        logger.info("Extracting sprint grid-based features...")
        if not sprint_results.empty:
            # Add grid position from SQ to sprint results for grid feature extraction
            sprint_with_grid = sprint_results.merge(
                sprint_quali_results[["year", "round", "driver_code", "position"]].rename(
                    columns={"position": "grid_position"}
                ),
                on=["year", "round", "driver_code"],
                how="left",
            )
            grid_features = self.grid_extractor.extract_features(sprint_with_grid)
            if not grid_features.empty:
                df = self._merge_features(df, grid_features, "grid")

        # FP1-only practice features
        logger.info("Extracting FP1-only practice features...")
        df = self._add_fp1_only_features(df, practice_sessions)

        # Historical sprint features
        logger.info("Extracting historical sprint features...")
        sprint_features = self.sprint_extractor.extract_features(
            sprint_results, sprint_quali_results, qualifying_results
        )
        if not sprint_features.empty:
            df = self._merge_features(df, sprint_features, "sprint")

        # Historical sector features from main qualifying (relevant to sprint pace)
        logger.info("Extracting historical sector features...")
        sector_features = self.sector_extractor.extract_features(quali_sessions)
        if not sector_features.empty:
            df = self._merge_features(df, sector_features, "sector")

        # Weather features (from FP1 only)
        logger.info("Extracting weather features...")
        fp1_sessions = practice_sessions[practice_sessions["session_type"] == "FP1"]
        if not fp1_sessions.empty:
            weather_features = self.weather_extractor.extract_features(fp1_sessions, fp1_sessions)
            if not weather_features.empty:
                df = self._merge_features(df, weather_features, "weather")

        return df

    def _add_sprint_grid_features(
        self,
        df: pd.DataFrame,
        sprint_quali_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add current sprint qualifying position as grid features.

        Sprint qualifying sets the grid for the sprint race.
        This is the most important feature for sprint race prediction.
        """
        if sprint_quali_results.empty:
            return df

        # Build lookup from SQ results
        sq_lookup = sprint_quali_results[["year", "round", "driver_code", "position"]].copy()
        sq_lookup = sq_lookup.rename(columns={"position": "sq_position"})
        sq_lookup = sq_lookup.drop_duplicates(subset=["year", "round", "driver_code"])

        # Merge to get SQ position for each sprint race entry
        df = df.merge(
            sq_lookup,
            on=["year", "round", "driver_code"],
            how="left",
        )

        # Create sprint grid features (SQ position = sprint race grid)
        df["sprint_grid_position"] = df["sq_position"]
        df["sprint_grid_pole"] = (df["sq_position"] == 1).astype(int)
        df["sprint_grid_front_row"] = (df["sq_position"] <= 2).astype(int)
        df["sprint_grid_top3"] = (df["sq_position"] <= 3).astype(int)
        df["sprint_grid_top5"] = (df["sq_position"] <= 5).astype(int)
        df["sprint_grid_top10"] = (df["sq_position"] <= 10).astype(int)

        # Calculate grid percentile
        df["sprint_grid_percentile"] = df.groupby(["year", "round"])["sq_position"].transform(
            lambda x: 1 - (x - 1) / (x.max() - 1) if x.max() > 1 else 1
        )

        # Drop temporary column
        df = df.drop(columns=["sq_position"], errors="ignore")

        return df

    def _add_fp1_only_features(
        self,
        df: pd.DataFrame,
        practice_sessions: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add features from FP1 only.

        FP2 and FP3 happen AFTER sprint race, so we cannot use them.
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

    def _get_feature_columns(self, windows: list[int]) -> list[str]:
        """Get all feature columns including sprint race-specific ones."""
        # Start with base features
        feature_cols = super()._get_feature_columns(windows)

        # CRITICAL: Sprint grid position features (from SQ)
        feature_cols.extend(
            [
                "sprint_grid_position",
                "sprint_grid_pole",
                "sprint_grid_front_row",
                "sprint_grid_top3",
                "sprint_grid_top5",
                "sprint_grid_top10",
                "sprint_grid_percentile",
            ]
        )

        # Historical sprint grid-to-finish features
        for w in windows:
            feature_cols.extend(
                [
                    f"avg_positions_gained_{w}",
                    f"positions_gained_std_{w}",
                    f"gain_rate_{w}",
                    f"loss_rate_{w}",
                    f"front_row_podium_rate_{w}",
                    f"pole_win_rate_{w}",
                    f"avg_grid_position_{w}",
                    f"avg_race_position_{w}",
                    f"grid_finish_delta_{w}",
                ]
            )
        feature_cols.append("first_lap_specialist")

        # FP1-only features
        feature_cols.extend(
            [
                "current_fp1_best_ms",
                "current_fp1_laps",
                "current_fp1_rank",
                "current_fp1_gap_ms",
                "current_fp1_gap_pct",
            ]
        )

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

    pipeline = SprintRaceFeaturePipeline()
    X, y, meta = pipeline.build_features(min_year=2021)

    print(f"\nFeature matrix shape: {X.shape}")
    print("\nSprint grid features:")
    grid_cols = [c for c in X.columns if "sprint_grid" in c.lower()]
    print(grid_cols)
    print("\nNOTE: No FP2/FP3 features - they happen after sprint race")
    print("\nTarget distribution (Sprint race positions):")
    print(y.value_counts().head(10))
