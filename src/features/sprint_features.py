"""
Sprint feature extraction for F1 predictions.

Extracts features from Sprint Qualifying (SQ) and Sprint Race (S) sessions.
Sprint weekends (2021+) provide additional predictive signals for main qualifying.

All rolling features use .shift(1) to prevent data leakage.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class SprintFeatureExtractor:
    """Extracts features from Sprint Qualifying and Sprint Race sessions."""

    def __init__(self, windows: list[int] | None = None):
        """
        Initialize sprint feature extractor.

        Args:
            windows: Rolling window sizes for temporal features (default: [3, 5])
        """
        self.windows = windows or [3, 5]
        logger.info(f"Initialized SprintFeatureExtractor (windows={self.windows})")

    def extract_features(
        self,
        sprint_results: pd.DataFrame,
        sprint_quali_results: pd.DataFrame,
        quali_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Extract sprint-related features for each driver/session.

        Args:
            sprint_results: DataFrame with sprint race results from loader
            sprint_quali_results: DataFrame with sprint qualifying results from loader
            quali_results: DataFrame with main qualifying results (for correlation)

        Returns:
            DataFrame with sprint features per driver per session
        """
        logger.info("Extracting sprint features...")

        if quali_results.empty:
            logger.warning("No qualifying results provided")
            return pd.DataFrame()

        # Start with qualifying sessions as base (we predict qualifying)
        features = quali_results[["session_key", "driver_code", "year", "round", "circuit"]].copy()

        # Add sprint weekend flag
        features = self._add_sprint_weekend_flag(features, sprint_results, sprint_quali_results)

        # Add sprint race features (historical)
        if not sprint_results.empty:
            features = self._add_sprint_race_features(features, sprint_results)

        # Add sprint qualifying features (historical)
        if not sprint_quali_results.empty:
            features = self._add_sprint_quali_features(features, sprint_quali_results)

        # Add cross-session features (sprint vs main quali correlation)
        if not sprint_quali_results.empty and not quali_results.empty:
            features = self._add_cross_session_features(
                features, sprint_quali_results, quali_results
            )

        logger.info(f"Extracted {len(features.columns) - 5} sprint features")
        return features

    def _add_sprint_weekend_flag(
        self,
        features: pd.DataFrame,
        sprint_results: pd.DataFrame,
        sprint_quali_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add binary flag indicating if this is a sprint weekend."""
        # Get all rounds that have sprint sessions
        sprint_rounds = set()

        if not sprint_results.empty:
            for _, row in sprint_results[["year", "round"]].drop_duplicates().iterrows():
                sprint_rounds.add((row["year"], row["round"]))

        if not sprint_quali_results.empty:
            for _, row in sprint_quali_results[["year", "round"]].drop_duplicates().iterrows():
                sprint_rounds.add((row["year"], row["round"]))

        # Add flag
        features["is_sprint_weekend"] = features.apply(
            lambda x: 1 if (x["year"], x["round"]) in sprint_rounds else 0, axis=1
        )

        return features

    def _add_sprint_race_features(
        self, features: pd.DataFrame, sprint_results: pd.DataFrame
    ) -> pd.DataFrame:
        """Add rolling features from sprint race results."""
        if sprint_results.empty:
            return features

        # Prepare sprint data for merging
        sprint_df = sprint_results[["driver_code", "year", "round", "position"]].copy()
        sprint_df = sprint_df.rename(columns={"position": "sprint_position"})

        # Add grid position if available for positions gained calculation
        if "grid_position" in sprint_results.columns:
            sprint_df["sprint_grid"] = sprint_results["grid_position"]
            sprint_df["sprint_positions_gained"] = (
                sprint_df["sprint_grid"] - sprint_df["sprint_position"]
            )

        # Sort for proper rolling calculations
        sprint_df = sprint_df.sort_values(["driver_code", "year", "round"])

        # Calculate rolling features per driver
        for window in self.windows:
            # Rolling average sprint position (shifted to exclude current)
            sprint_df[f"sprint_avg_position_{window}"] = sprint_df.groupby("driver_code")[
                "sprint_position"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

            # Rolling sprint consistency (std dev of positions)
            sprint_df[f"sprint_consistency_{window}"] = sprint_df.groupby("driver_code")[
                "sprint_position"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=2).std())

            # Rolling positions gained in sprints
            if "sprint_positions_gained" in sprint_df.columns:
                sprint_df[f"sprint_positions_gained_{window}"] = sprint_df.groupby("driver_code")[
                    "sprint_positions_gained"
                ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

        # Merge sprint features back to main features
        # Need to align by year/round - sprint data from PRIOR sprints affects CURRENT race weekend
        sprint_feature_cols = [
            c
            for c in sprint_df.columns
            if c.startswith("sprint_avg")
            or c.startswith("sprint_consistency")
            or c.startswith("sprint_positions")
        ]

        if sprint_feature_cols:
            sprint_features = sprint_df[["driver_code", "year", "round"] + sprint_feature_cols]
            features = features.merge(
                sprint_features,
                on=["driver_code", "year", "round"],
                how="left",
            )

        return features

    def _add_sprint_quali_features(
        self, features: pd.DataFrame, sprint_quali_results: pd.DataFrame
    ) -> pd.DataFrame:
        """Add rolling features from sprint qualifying results."""
        if sprint_quali_results.empty:
            return features

        # Prepare sprint qualifying data
        sq_df = sprint_quali_results[["driver_code", "year", "round", "position"]].copy()
        sq_df = sq_df.rename(columns={"position": "sq_position"})

        # Sort for proper rolling calculations
        sq_df = sq_df.sort_values(["driver_code", "year", "round"])

        # Calculate rolling features per driver
        for window in self.windows:
            # Rolling average sprint qualifying position
            sq_df[f"sq_avg_position_{window}"] = sq_df.groupby("driver_code")[
                "sq_position"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

            # Rolling sprint qualifying consistency
            sq_df[f"sq_consistency_{window}"] = sq_df.groupby("driver_code")[
                "sq_position"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=2).std())

        # Merge sprint qualifying features
        sq_feature_cols = [
            c for c in sq_df.columns if c.startswith("sq_avg") or c.startswith("sq_consistency")
        ]

        if sq_feature_cols:
            sq_features = sq_df[["driver_code", "year", "round"] + sq_feature_cols]
            features = features.merge(
                sq_features,
                on=["driver_code", "year", "round"],
                how="left",
            )

        return features

    def _add_cross_session_features(
        self,
        features: pd.DataFrame,
        sprint_quali_results: pd.DataFrame,
        quali_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add features comparing sprint qualifying to main qualifying performance."""
        if sprint_quali_results.empty or quali_results.empty:
            return features

        # Merge SQ and Q results for same driver/round
        sq_df = sprint_quali_results[["driver_code", "year", "round", "position"]].copy()
        sq_df = sq_df.rename(columns={"position": "sq_position"})

        q_df = quali_results[["driver_code", "year", "round", "position"]].copy()
        q_df = q_df.rename(columns={"position": "q_position"})

        merged = sq_df.merge(q_df, on=["driver_code", "year", "round"], how="inner")

        if merged.empty:
            return features

        # Calculate SQ vs Q position delta (positive = better in main Q than SQ)
        merged["sq_to_q_delta"] = merged["sq_position"] - merged["q_position"]

        # Sort for rolling
        merged = merged.sort_values(["driver_code", "year", "round"])

        # Calculate rolling average of SQ to Q improvement
        for window in self.windows:
            merged[f"sq_to_q_improvement_{window}"] = merged.groupby("driver_code")[
                "sq_to_q_delta"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

        # Merge back
        cross_cols = [c for c in merged.columns if c.startswith("sq_to_q_improvement")]

        if cross_cols:
            cross_features = merged[["driver_code", "year", "round"] + cross_cols]
            features = features.merge(
                cross_features,
                on=["driver_code", "year", "round"],
                how="left",
            )

        return features

    def get_feature_names(self) -> list[str]:
        """Get list of feature names this extractor produces."""
        names = ["is_sprint_weekend"]

        for window in self.windows:
            # Sprint race features
            names.extend(
                [
                    f"sprint_avg_position_{window}",
                    f"sprint_consistency_{window}",
                    f"sprint_positions_gained_{window}",
                ]
            )
            # Sprint qualifying features
            names.extend(
                [
                    f"sq_avg_position_{window}",
                    f"sq_consistency_{window}",
                ]
            )
            # Cross-session features
            names.append(f"sq_to_q_improvement_{window}")

        return names


def main():
    """Command-line interface for sprint feature extraction."""
    import argparse
    from pathlib import Path

    from src.data.loaders import F1DataLoader

    parser = argparse.ArgumentParser(description="Extract sprint features")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/fastf1",
        help="Data directory (default: data/fastf1)",
    )
    parser.add_argument("--min-year", type=int, default=2021, help="Minimum year (default: 2021)")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load data
    loader = F1DataLoader(Path(args.data_dir))

    quali_results = loader.load_qualifying_results(min_year=args.min_year)
    sprint_results = loader.load_sprint_results(min_year=args.min_year)
    sprint_quali_results = loader.load_sprint_qualifying_results(min_year=args.min_year)

    print("\nLoaded data:")
    print(f"  Qualifying sessions: {len(quali_results)}")
    print(f"  Sprint races: {len(sprint_results)}")
    print(f"  Sprint qualifying: {len(sprint_quali_results)}")

    # Extract features
    extractor = SprintFeatureExtractor(windows=[3, 5])
    features = extractor.extract_features(sprint_results, sprint_quali_results, quali_results)

    print(f"\nExtracted features: {features.shape}")
    print(
        f"Feature columns: {[c for c in features.columns if c not in ['session_key', 'driver_code', 'year', 'round', 'circuit']]}"
    )

    # Show sample
    sprint_weekends = features[features["is_sprint_weekend"] == 1]
    print(f"\nSprint weekend samples: {len(sprint_weekends)}")
    if not sprint_weekends.empty:
        print(sprint_weekends.head(10))


if __name__ == "__main__":
    main()
