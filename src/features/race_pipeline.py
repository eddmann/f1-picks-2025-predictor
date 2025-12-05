"""
Race feature pipeline for F1 predictions.

Extends BaseFeaturePipeline with race-specific features:
- Qualifying grid position (most important feature ~60-70% variance)
- Sprint results (if sprint weekend)
- All practice session features (FP1, FP2, FP3)
- Race-specific temporal features (positions gained, pit strategy)

Temporal constraints:
- Can use: FP1, FP2, FP3, Q results (grid), SQ results, S results, historical
- Cannot use: Current race results (that's the target)
"""

import logging
from pathlib import Path

import pandas as pd

from src.features.base_pipeline import BaseFeaturePipeline
from src.features.grid_features import GridPositionFeatureExtractor
from src.features.practice_features import PracticeFeatureExtractor
from src.features.race_pace_features import RacePaceFeatureExtractor

logger = logging.getLogger(__name__)


class RaceFeaturePipeline(BaseFeaturePipeline):
    """
    Feature pipeline for main race predictions.

    Uses all available data from the weekend:
    - All practice sessions (FP1, FP2, FP3)
    - Qualifying results as grid position (most important feature)
    - Sprint qualifying and sprint race results (if sprint weekend)
    - Historical race performance features

    Grid position is the strongest predictor of race finish position.
    """

    session_type = "R"

    def __init__(self, data_dir: Path | str = "data/fastf1"):
        """
        Initialize the race feature pipeline.

        Args:
            data_dir: Directory containing FastF1 parquet files
        """
        super().__init__(data_dir)

        # Initialize race-specific extractors
        self.grid_extractor = GridPositionFeatureExtractor()
        self.practice_extractor = PracticeFeatureExtractor()
        self.race_pace_extractor = RacePaceFeatureExtractor()

    def get_available_practice_sessions(self) -> list[str]:
        """All practice sessions are available before the race."""
        return ["FP1", "FP2", "FP3"]

    def get_available_current_weekend_sessions(self) -> list[str]:
        """Qualifying and sprint sessions are available before the race."""
        return ["Q", "SQ", "S"]

    def _load_target_results(self, min_year: int) -> pd.DataFrame:
        """Load race results as the prediction target."""
        return self.loader.load_race_results(min_year=min_year)

    def _add_first_lap_features(
        self,
        df: pd.DataFrame,
        target_results: pd.DataFrame,
        min_year: int,
    ) -> pd.DataFrame:
        """
        Skip first lap features for race pipeline.

        First lap position data is not available in our data (the 'position' column
        contains final race position, not lap-by-lap positions). The race_pace_features
        already captures positions gained/lost which would be identical.
        """
        logger.info("Skipping first lap features for race (covered by race_pace_features)")
        return df

    def _get_session_specific_features(
        self,
        df: pd.DataFrame,
        min_year: int,
        windows: list[int],
    ) -> pd.DataFrame:
        """
        Add race-specific features.

        Includes:
        - Current weekend qualifying grid position (CRUCIAL)
        - Historical grid-to-finish performance
        - Practice session features (FP1-FP3)
        - Sprint features (if sprint weekend)
        - Weather features
        - Sector/tyre features from qualifying
        """
        # Load session data
        race_results = self._load_target_results(min_year=min_year)
        qualifying_results = self.loader.load_qualifying_results(min_year=min_year)
        practice_sessions = self.loader.load_practice_sessions(min_year=min_year)
        quali_sessions = self.loader.load_qualifying_sessions(min_year=min_year)

        # Load sprint data (2021+)
        sprint_results = self.loader.load_sprint_results(min_year=max(min_year, 2021))
        sprint_quali_results = self.loader.load_sprint_qualifying_results(
            min_year=max(min_year, 2021)
        )

        # CRITICAL: Add current qualifying grid position
        logger.info("Adding current grid position features...")
        df = self._add_current_grid_features(df, qualifying_results)

        # Historical grid-to-finish performance
        logger.info("Extracting grid-based features...")
        grid_features = self.grid_extractor.extract_features(race_results)
        if not grid_features.empty:
            df = self._merge_features(df, grid_features, "grid")

        # Race pace features (race specialist identification, DNF rate, etc.)
        logger.info("Extracting race pace features...")
        race_pace_features = self.race_pace_extractor.extract_features(race_results)
        if not race_pace_features.empty:
            df = self._merge_features(df, race_pace_features, "race_pace")

        # Practice features (FP1/FP2/FP3)
        logger.info("Extracting practice features...")
        practice_features = self.practice_extractor.extract_features(
            practice_sessions, quali_sessions
        )
        if not practice_features.empty:
            df = self._merge_features(df, practice_features, "practice")

        # Sector features from qualifying (relevant to race pace)
        logger.info("Extracting sector features...")
        sector_features = self.sector_extractor.extract_features(quali_sessions)
        if not sector_features.empty:
            df = self._merge_features(df, sector_features, "sector")

        # Tyre features (crucial for race strategy)
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

        # Add current sprint results as features (for sprint weekends)
        df = self._add_current_sprint_features(df, sprint_results, sprint_quali_results)

        # Weather features
        logger.info("Extracting weather features...")
        weather_features = self.weather_extractor.extract_features(
            quali_sessions, practice_sessions
        )
        if not weather_features.empty:
            df = self._merge_features(df, weather_features, "weather")

        return df

    def _add_current_grid_features(
        self,
        df: pd.DataFrame,
        qualifying_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add current qualifying position as grid features.

        This is the most important feature for race prediction.
        """
        if qualifying_results.empty:
            return df

        # Build lookup from qualifying results
        quali_lookup = qualifying_results[["year", "round", "driver_code", "position"]].copy()
        quali_lookup = quali_lookup.rename(columns={"position": "quali_position"})
        quali_lookup = quali_lookup.drop_duplicates(subset=["year", "round", "driver_code"])

        # Merge to get qualifying position for each race entry
        df = df.merge(
            quali_lookup,
            on=["year", "round", "driver_code"],
            how="left",
        )

        # Create grid position features
        df["current_grid_position"] = df["quali_position"]
        df["current_grid_pole"] = (df["quali_position"] == 1).astype(int)
        df["current_grid_front_row"] = (df["quali_position"] <= 2).astype(int)
        df["current_grid_top3"] = (df["quali_position"] <= 3).astype(int)
        df["current_grid_top5"] = (df["quali_position"] <= 5).astype(int)
        df["current_grid_top10"] = (df["quali_position"] <= 10).astype(int)

        # Calculate grid percentile (higher = better starting position)
        # For each race, calculate relative grid position
        df["current_grid_percentile"] = df.groupby(["year", "round"])["quali_position"].transform(
            lambda x: 1 - (x - 1) / (x.max() - 1) if x.max() > 1 else 1
        )

        # Drop temporary column
        df = df.drop(columns=["quali_position"], errors="ignore")

        return df

    def _add_current_sprint_features(
        self,
        df: pd.DataFrame,
        sprint_results: pd.DataFrame,
        sprint_quali_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add current weekend sprint results as features.

        Sprint results are available before the main race on sprint weekends.
        """
        # Initialize is_sprint_weekend to 0 for all rows first
        df["is_sprint_weekend"] = 0

        if sprint_results.empty and sprint_quali_results.empty:
            return df

        # Determine sprint weekends from either source
        sprint_year_rounds = set()
        if not sprint_results.empty:
            for _, row in sprint_results[["year", "round"]].drop_duplicates().iterrows():
                sprint_year_rounds.add((row["year"], row["round"]))
        if not sprint_quali_results.empty:
            for _, row in sprint_quali_results[["year", "round"]].drop_duplicates().iterrows():
                sprint_year_rounds.add((row["year"], row["round"]))

        # Mark sprint weekends
        if sprint_year_rounds:
            df["is_sprint_weekend"] = df.apply(
                lambda row: 1 if (row["year"], row["round"]) in sprint_year_rounds else 0, axis=1
            )

        # Add current sprint race position
        if not sprint_results.empty:
            sprint_lookup = sprint_results[["year", "round", "driver_code", "position"]].copy()
            sprint_lookup = sprint_lookup.rename(columns={"position": "current_sprint_position"})
            sprint_lookup = sprint_lookup.drop_duplicates(subset=["year", "round", "driver_code"])

            df = df.merge(
                sprint_lookup,
                on=["year", "round", "driver_code"],
                how="left",
            )

            # Create sprint position features
            df["current_sprint_top3"] = (
                (df["current_sprint_position"] <= 3) & (df["is_sprint_weekend"] == 1)
            ).astype(int)
            df["current_sprint_top5"] = (
                (df["current_sprint_position"] <= 5) & (df["is_sprint_weekend"] == 1)
            ).astype(int)

        # Add sprint qualifying position
        if not sprint_quali_results.empty:
            sq_lookup = sprint_quali_results[["year", "round", "driver_code", "position"]].copy()
            sq_lookup = sq_lookup.rename(columns={"position": "current_sq_position"})
            sq_lookup = sq_lookup.drop_duplicates(subset=["year", "round", "driver_code"])

            df = df.merge(
                sq_lookup,
                on=["year", "round", "driver_code"],
                how="left",
            )

        # Fill missing sprint features with 0 for non-sprint weekends
        sprint_cols = [
            "current_sprint_position",
            "current_sprint_top3",
            "current_sprint_top5",
            "current_sq_position",
        ]
        for col in sprint_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        return df

    def _get_feature_columns(self, windows: list[int]) -> list[str]:
        """Get all feature columns including race-specific ones."""
        # Start with base features, then remove first lap features
        # (they duplicate race_pace_features due to data limitations)
        feature_cols = super()._get_feature_columns(windows)
        first_lap_cols = [c for c in feature_cols if "first_lap" in c or c.startswith("start_")]
        feature_cols = [c for c in feature_cols if c not in first_lap_cols]

        # CRITICAL: Grid position features
        feature_cols.extend(
            [
                "current_grid_position",
                "current_grid_pole",
                "current_grid_front_row",
                "current_grid_top3",
                "current_grid_top5",
                "current_grid_top10",
                "current_grid_percentile",
            ]
        )

        # Historical grid-to-finish features
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

        # Race pace features (race specialist identification)
        for w in windows:
            feature_cols.extend(
                [
                    f"max_positions_gained_{w}",
                    f"dnf_rate_{w}",
                    f"finish_rate_{w}",
                    f"team_dnf_rate_{w}",
                ]
            )
        feature_cols.extend(
            [
                "race_specialist",
                "quali_specialist",
                "consistent_gainer",
                "consistent_loser",
                "reliable_driver",
                "unreliable_driver",
            ]
        )

        # Practice features (current weekend)
        feature_cols.extend(
            [
                "current_fp1_best_ms",
                "current_fp2_best_ms",
                "current_fp3_best_ms",
                "current_practice_best_ms",
                "current_total_practice_laps",
                "current_fp1_to_fp3_improvement_pct",
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

        # Sector features
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

        # Tyre features (important for race strategy)
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

        # Sprint features
        feature_cols.extend(
            [
                "is_sprint_weekend",
                "current_sprint_position",
                "current_sprint_top3",
                "current_sprint_top5",
                "current_sq_position",
            ]
        )
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

        # Weather features
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

    pipeline = RaceFeaturePipeline()
    X, y, meta = pipeline.build_features(min_year=2024)

    print(f"\nFeature matrix shape: {X.shape}")
    print("\nTop features (grid-related):")
    grid_cols = [c for c in X.columns if "grid" in c.lower()]
    print(grid_cols)
    print("\nTarget distribution (race positions):")
    print(y.value_counts().head(10))
