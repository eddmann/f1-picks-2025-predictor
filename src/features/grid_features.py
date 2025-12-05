"""
Grid position feature extraction.

Extracts features based on grid positions for race/sprint predictions.
Grid position is the most predictive feature for race outcomes (~60-70% variance).

All features use temporal safety patterns (shift before rolling) to prevent data leakage.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class GridPositionFeatureExtractor:
    """
    Extracts grid position features for race and sprint race predictions.

    These features capture:
    - Historical grid-to-finish performance
    - Positions gained/lost patterns
    - Front row conversion rates
    - First lap performance trends
    """

    def __init__(self, windows: list[int] | None = None):
        """
        Initialize grid position feature extractor.

        Args:
            windows: Rolling window sizes for aggregations (default: [3, 5, 10])
        """
        self.windows = windows or [3, 5, 10]

    def extract_features(
        self,
        race_results: pd.DataFrame,
        grid_source: str = "qualifying",
    ) -> pd.DataFrame:
        """
        Extract grid-to-finish performance features.

        Args:
            race_results: DataFrame with race/sprint results including grid_position
            grid_source: Source of grid ("qualifying" for races, "sprint_quali" for sprints)

        Returns:
            DataFrame with grid-based features per driver per session
        """
        if race_results.empty:
            return pd.DataFrame()

        required_cols = ["session_key", "driver_code", "position", "grid_position"]
        missing = [c for c in required_cols if c not in race_results.columns]
        if missing:
            logger.warning(f"Missing columns for grid features: {missing}")
            return pd.DataFrame()

        # Compute session-level stats
        session_features = self._compute_session_grid_stats(race_results)

        if session_features.empty:
            return pd.DataFrame()

        # Compute rolling features
        rolling_features = self._compute_rolling_grid_features(session_features)

        return rolling_features

    def _compute_session_grid_stats(
        self,
        race_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute grid-to-finish statistics for each driver per session."""
        df = race_results.copy()

        # Ensure numeric columns
        df["position"] = pd.to_numeric(df["position"], errors="coerce")
        df["grid_position"] = pd.to_numeric(df["grid_position"], errors="coerce")

        # Calculate positions gained (positive = gained, negative = lost)
        df["positions_gained"] = df["grid_position"] - df["position"]

        # Binary indicators
        df["gained_positions"] = (df["positions_gained"] > 0).astype(int)
        df["lost_positions"] = (df["positions_gained"] < 0).astype(int)
        df["held_position"] = (df["positions_gained"] == 0).astype(int)

        # Grid position categories
        df["grid_pole"] = (df["grid_position"] == 1).astype(int)
        df["grid_front_row"] = (df["grid_position"] <= 2).astype(int)
        df["grid_top3"] = (df["grid_position"] <= 3).astype(int)
        df["grid_top5"] = (df["grid_position"] <= 5).astype(int)
        df["grid_top10"] = (df["grid_position"] <= 10).astype(int)

        # Finish indicators
        df["finish_top3"] = (df["position"] <= 3).astype(int)
        df["finish_podium_from_front_row"] = (
            (df["grid_position"] <= 2) & (df["position"] <= 3)
        ).astype(int)
        df["finish_podium_from_grid_top3"] = (
            (df["grid_position"] <= 3) & (df["position"] <= 3)
        ).astype(int)

        # Grid to finish conversion
        df["pole_to_win"] = ((df["grid_position"] == 1) & (df["position"] == 1)).astype(int)
        df["front_row_to_podium"] = ((df["grid_position"] <= 2) & (df["position"] <= 3)).astype(int)

        # Select columns to keep
        keep_cols = [
            "session_key",
            "driver_code",
            "year",
            "round",
            "circuit",
            "team",
            "position",
            "grid_position",
            "positions_gained",
            "gained_positions",
            "lost_positions",
            "held_position",
            "grid_pole",
            "grid_front_row",
            "grid_top3",
            "grid_top5",
            "grid_top10",
            "finish_top3",
            "finish_podium_from_front_row",
            "finish_podium_from_grid_top3",
            "pole_to_win",
            "front_row_to_podium",
        ]

        keep_cols = [c for c in keep_cols if c in df.columns]
        return df[keep_cols].drop_duplicates(subset=["session_key", "driver_code"])

    def _compute_rolling_grid_features(
        self,
        session_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute rolling features across sessions with temporal safety."""
        df = session_features.copy()
        df = df.sort_values(["driver_code", "year", "round"])

        for window in self.windows:
            # Rolling average positions gained
            df[f"avg_positions_gained_{window}"] = df.groupby("driver_code")[
                "positions_gained"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

            # Rolling std of positions gained (consistency)
            df[f"positions_gained_std_{window}"] = df.groupby("driver_code")[
                "positions_gained"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=2).std())

            # Rate of gaining positions
            df[f"gain_rate_{window}"] = df.groupby("driver_code")["gained_positions"].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

            # Rate of losing positions
            df[f"loss_rate_{window}"] = df.groupby("driver_code")["lost_positions"].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

            # Front row to podium conversion rate
            df[f"front_row_podium_rate_{window}"] = df.groupby("driver_code")[
                "front_row_to_podium"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

            # Pole to win conversion rate
            df[f"pole_win_rate_{window}"] = df.groupby("driver_code")["pole_to_win"].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

            # Average grid position
            df[f"avg_grid_position_{window}"] = df.groupby("driver_code")[
                "grid_position"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

            # Average finish position (from races, not qualifying)
            df[f"avg_race_position_{window}"] = df.groupby("driver_code")["position"].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

            # Grid vs finish delta (positive = consistently gaining)
            df[f"grid_finish_delta_{window}"] = (
                df[f"avg_grid_position_{window}"] - df[f"avg_race_position_{window}"]
            )

        # First lap specialist indicator (do they typically gain on lap 1?)
        if 5 in self.windows:
            df["first_lap_specialist"] = (df["avg_positions_gained_5"] > 1).astype(int)

        return df

    def extract_current_grid_features(
        self,
        grid_position: int,
        field_size: int = 20,
    ) -> dict[str, float]:
        """
        Extract features from a driver's current grid position.

        Used at prediction time when we have the actual grid position.

        Args:
            grid_position: Driver's grid position (1-20)
            field_size: Total number of drivers in the race

        Returns:
            Dictionary of grid position features
        """
        return {
            "current_grid_position": grid_position,
            "current_grid_pole": 1 if grid_position == 1 else 0,
            "current_grid_front_row": 1 if grid_position <= 2 else 0,
            "current_grid_top3": 1 if grid_position <= 3 else 0,
            "current_grid_top5": 1 if grid_position <= 5 else 0,
            "current_grid_top10": 1 if grid_position <= 10 else 0,
            "current_grid_percentile": (field_size - grid_position + 1) / field_size,
            "current_grid_normalized": (field_size - grid_position) / (field_size - 1),
        }


def add_current_grid_to_features(
    X: pd.DataFrame,
    qualifying_results: pd.DataFrame,
    meta: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add current qualifying grid positions as features for race prediction.

    This is used at training time to merge qualifying results with race features.

    Args:
        X: Feature matrix for race prediction
        qualifying_results: Qualifying results with position (used as grid)
        meta: Metadata with session identifiers

    Returns:
        Feature matrix with grid position features added
    """
    X = X.copy()

    # Build lookup from qualifying results
    quali_lookup = qualifying_results[["year", "round", "driver_code", "position"]].copy()
    quali_lookup = quali_lookup.rename(columns={"position": "grid_position"})

    # Merge with meta to get year/round
    if "year" in meta.columns and "round" in meta.columns:
        merge_data = meta[["year", "round", "driver_code"]].copy()
        merge_data.index = X.index

        merged = merge_data.merge(
            quali_lookup,
            on=["year", "round", "driver_code"],
            how="left",
        )

        # Add grid position features
        extractor = GridPositionFeatureExtractor()
        for idx, row in merged.iterrows():
            if pd.notna(row["grid_position"]):
                grid_features = extractor.extract_current_grid_features(int(row["grid_position"]))
                for feat_name, feat_val in grid_features.items():
                    if feat_name not in X.columns:
                        X[feat_name] = 0.0
                    X.loc[idx, feat_name] = feat_val

    return X
