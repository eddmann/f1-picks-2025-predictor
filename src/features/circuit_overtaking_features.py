"""
Circuit overtaking difficulty feature extraction for F1 predictions.

Some circuits are notoriously difficult to overtake on (Monaco, Hungary),
while others promote lots of position changes (Monza, Bahrain). This affects:
- How much grid position matters (more at low-overtaking circuits)
- Whether fast cars can recover from poor qualifying
- Strategy effectiveness (undercut vs overcut)

Key insights from research:
- Monaco has ~10% overtaking rate, Monza has ~40%+
- DRS zones and straight length correlate with overtaking
- Historical position changes are the best proxy for overtaking difficulty

All features use temporal shift to prevent data leakage.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CircuitOvertakingExtractor:
    """
    Extract circuit overtaking difficulty features for F1 prediction.

    Uses historical position change data to quantify how easy/hard it is
    to overtake at each circuit. Also creates driver-specific features
    for overtaking ability.
    """

    def __init__(
        self,
        windows: list[int] | None = None,
        data_dir: Path | str = "data/fastf1",
    ):
        """
        Initialize circuit overtaking extractor.

        Args:
            windows: Rolling window sizes for temporal features
            data_dir: Directory containing FastF1 parquet files
        """
        self.windows = windows or [3, 5, 10]
        self.data_dir = Path(data_dir)
        logger.info(f"Initialized CircuitOvertakingExtractor (windows={self.windows})")

    def _load_race_data(self, min_year: int) -> pd.DataFrame:
        """
        Load race session data with lap-by-lap positions.

        Args:
            min_year: Minimum year to load

        Returns:
            DataFrame with race lap data
        """
        sessions_dir = self.data_dir / "sessions"
        if not sessions_dir.exists():
            logger.warning(f"Sessions directory not found: {sessions_dir}")
            return pd.DataFrame()

        dfs = []
        for file_path in sessions_dir.glob("*_R.parquet"):
            parts = file_path.stem.split("_")
            if len(parts) >= 2:
                year = int(parts[0])
                if year >= min_year:
                    try:
                        df = pd.read_parquet(file_path)
                        dfs.append(df)
                    except Exception as e:
                        logger.warning(f"Failed to load {file_path}: {e}")

        if not dfs:
            return pd.DataFrame()

        all_data = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(all_data)} rows from race sessions")
        return all_data

    def _calculate_circuit_overtaking_stats(
        self,
        race_data: pd.DataFrame,
        race_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate overtaking statistics per circuit.

        Uses position changes between grid and finish as proxy for overtaking.

        Args:
            race_data: Raw race lap data
            race_results: Race results with grid and finish positions

        Returns:
            DataFrame with circuit overtaking stats
        """
        if race_results.empty:
            return pd.DataFrame()

        df = race_results.copy()

        # Ensure we have grid_position (from qualifying)
        if "grid_position" not in df.columns:
            # Try to infer from position column or use a placeholder
            logger.warning("No grid_position in race results, using position as fallback")
            df["grid_position"] = df["position"]

        # Calculate positions changed (absolute value)
        df["positions_changed"] = abs(
            pd.to_numeric(df["grid_position"], errors="coerce")
            - pd.to_numeric(df["position"], errors="coerce")
        )

        # Calculate positions gained (positive = gained, negative = lost)
        df["positions_gained"] = pd.to_numeric(
            df["grid_position"], errors="coerce"
        ) - pd.to_numeric(df["position"], errors="coerce")

        # Determine circuit column name
        circuit_col = "circuit_id" if "circuit_id" in df.columns else "circuit"
        if circuit_col not in df.columns:
            logger.warning("No circuit column found in race results")
            return pd.DataFrame()

        # Group by circuit to get overtaking stats
        circuit_stats = []

        for circuit in df[circuit_col].unique():
            circuit_df = df[df[circuit_col] == circuit].copy()

            # Sort by year/round for temporal ordering
            circuit_df = circuit_df.sort_values(["year", "round"])

            # Calculate rolling stats with shift to prevent leakage
            for _idx, (year, round_num) in enumerate(
                circuit_df[["year", "round"]].drop_duplicates().values
            ):
                # Get all prior races at this circuit
                prior_mask = (circuit_df["year"] < year) | (
                    (circuit_df["year"] == year) & (circuit_df["round"] < round_num)
                )
                prior_data = circuit_df[prior_mask]

                if prior_data.empty:
                    # No historical data for this circuit yet
                    stats = {
                        "circuit": circuit,
                        "year": year,
                        "round": round_num,
                        "circuit_avg_positions_changed": np.nan,
                        "circuit_max_positions_changed": np.nan,
                        "circuit_positions_changed_std": np.nan,
                        "circuit_overtaking_rate": np.nan,
                        "circuit_gain_rate": np.nan,
                        "circuit_loss_rate": np.nan,
                        "circuit_hold_rate": np.nan,
                        "circuit_race_count": 0,
                    }
                else:
                    # Calculate historical stats for this circuit
                    avg_changed = prior_data["positions_changed"].mean()
                    max_changed = prior_data["positions_changed"].max()
                    std_changed = prior_data["positions_changed"].std()

                    # Overtaking rate: % of drivers who gained positions
                    total_drivers = len(prior_data)
                    gained = (prior_data["positions_gained"] > 0).sum()
                    lost = (prior_data["positions_gained"] < 0).sum()
                    held = (prior_data["positions_gained"] == 0).sum()

                    stats = {
                        "circuit": circuit,
                        "year": year,
                        "round": round_num,
                        "circuit_avg_positions_changed": avg_changed,
                        "circuit_max_positions_changed": max_changed,
                        "circuit_positions_changed_std": std_changed,
                        "circuit_overtaking_rate": gained / total_drivers
                        if total_drivers > 0
                        else 0,
                        "circuit_gain_rate": gained / total_drivers if total_drivers > 0 else 0,
                        "circuit_loss_rate": lost / total_drivers if total_drivers > 0 else 0,
                        "circuit_hold_rate": held / total_drivers if total_drivers > 0 else 0,
                        "circuit_race_count": len(prior_data[["year", "round"]].drop_duplicates()),
                    }

                circuit_stats.append(stats)

        if not circuit_stats:
            return pd.DataFrame()

        return pd.DataFrame(circuit_stats)

    def _calculate_driver_overtaking_features(
        self,
        race_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate driver-specific overtaking ability features.

        Args:
            race_results: Race results with position changes

        Returns:
            DataFrame with driver overtaking features
        """
        df = race_results.copy()
        df = df.sort_values(["driver_code", "year", "round"])

        # Calculate positions gained
        if "grid_position" not in df.columns:
            df["grid_position"] = df["position"]

        df["positions_gained"] = pd.to_numeric(
            df["grid_position"], errors="coerce"
        ) - pd.to_numeric(df["position"], errors="coerce")

        features = df[["session_key", "driver_code", "year", "round"]].copy()

        for window in self.windows:
            # Driver's average positions gained (overtaking ability)
            features[f"driver_avg_overtakes_{window}"] = df.groupby("driver_code")[
                "positions_gained"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

            # Driver's overtaking consistency
            features[f"driver_overtakes_std_{window}"] = df.groupby("driver_code")[
                "positions_gained"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).std())

            # Rate of gaining positions
            features[f"driver_gain_rate_{window}"] = df.groupby("driver_code")[
                "positions_gained"
            ].transform(lambda x: (x.shift(1) > 0).rolling(window, min_periods=1).mean())

            # Max positions gained in window
            features[f"driver_max_overtakes_{window}"] = df.groupby("driver_code")[
                "positions_gained"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).max())

        # Career overtaking stats
        features["driver_career_avg_overtakes"] = df.groupby("driver_code")[
            "positions_gained"
        ].transform(lambda x: x.shift(1).expanding().mean())

        features["driver_career_overtake_rate"] = df.groupby("driver_code")[
            "positions_gained"
        ].transform(lambda x: (x.shift(1) > 0).expanding().mean())

        # Binary flags for overtaking specialists
        features["overtaking_specialist"] = (
            (features["driver_career_avg_overtakes"] > 1.0)
            & (features["driver_career_overtake_rate"] > 0.5)
        ).astype(float)

        # Position defender (rarely loses positions)
        features["position_defender"] = (
            (features["driver_career_avg_overtakes"] > -0.5)
            & (features["driver_career_overtake_rate"] > 0.4)
        ).astype(float)

        return features

    def extract_features(
        self,
        target_results: pd.DataFrame,
        min_year: int = 2020,
    ) -> pd.DataFrame:
        """
        Extract circuit overtaking features.

        Args:
            target_results: Target session results (for merging context)
            min_year: Minimum year for loading data

        Returns:
            DataFrame with circuit overtaking features per driver per session
        """
        logger.info("Extracting circuit overtaking features...")

        # Load race data for position changes
        race_data = self._load_race_data(min_year)

        # Get circuit stats
        circuit_stats = self._calculate_circuit_overtaking_stats(race_data, target_results)

        # Get driver overtaking features
        driver_features = self._calculate_driver_overtaking_features(target_results)

        # Merge circuit stats with target results
        if not circuit_stats.empty:
            # Use correct circuit column name
            circuit_col = "circuit_id" if "circuit_id" in target_results.columns else "circuit"
            merge_cols = [circuit_col, "year", "round"]
            # Rename circuit column in stats if needed
            if circuit_col == "circuit_id" and "circuit" in circuit_stats.columns:
                circuit_stats = circuit_stats.rename(columns={"circuit": "circuit_id"})
            df = target_results.merge(
                circuit_stats,
                on=merge_cols,
                how="left",
            )
        else:
            df = target_results.copy()
            # Add empty circuit columns
            for col in [
                "circuit_avg_positions_changed",
                "circuit_max_positions_changed",
                "circuit_positions_changed_std",
                "circuit_overtaking_rate",
                "circuit_gain_rate",
                "circuit_loss_rate",
                "circuit_hold_rate",
                "circuit_race_count",
            ]:
                df[col] = np.nan

        # Merge driver features
        if not driver_features.empty:
            df = df.merge(
                driver_features,
                on=["session_key", "driver_code", "year", "round"],
                how="left",
            )

        # Create circuit difficulty score (inverse of overtaking rate)
        # Higher = harder to overtake
        df["circuit_difficulty"] = 1 - df["circuit_overtaking_rate"].fillna(0.5)

        # Grid importance: how much grid position matters at this circuit
        # At low-overtaking circuits, grid is more important
        df["grid_importance"] = df["circuit_difficulty"]

        # Interaction features: driver overtaking ability * circuit difficulty
        for window in self.windows:
            if f"driver_avg_overtakes_{window}" in df.columns:
                df[f"overtake_potential_{window}"] = df[f"driver_avg_overtakes_{window}"] * df[
                    "circuit_overtaking_rate"
                ].fillna(0.5)

        # Categorize circuits by overtaking difficulty
        df["low_overtaking_circuit"] = (df["circuit_difficulty"] > 0.6).astype(float)
        df["high_overtaking_circuit"] = (df["circuit_difficulty"] < 0.4).astype(float)

        # Select feature columns
        feature_cols = [
            "session_key",
            "driver_code",
            "year",
            "round",
            "circuit_avg_positions_changed",
            "circuit_max_positions_changed",
            "circuit_positions_changed_std",
            "circuit_overtaking_rate",
            "circuit_difficulty",
            "grid_importance",
            "circuit_race_count",
            "low_overtaking_circuit",
            "high_overtaking_circuit",
        ]

        # Add driver features
        for window in self.windows:
            feature_cols.extend(
                [
                    f"driver_avg_overtakes_{window}",
                    f"driver_overtakes_std_{window}",
                    f"driver_gain_rate_{window}",
                    f"driver_max_overtakes_{window}",
                    f"overtake_potential_{window}",
                ]
            )

        feature_cols.extend(
            [
                "driver_career_avg_overtakes",
                "driver_career_overtake_rate",
                "overtaking_specialist",
                "position_defender",
            ]
        )

        # Only keep columns that exist
        feature_cols = [c for c in feature_cols if c in df.columns]
        features = df[feature_cols].copy()

        n_features = len(
            [
                c
                for c in features.columns
                if c not in ["session_key", "driver_code", "year", "round"]
            ]
        )
        logger.info(f"Extracted {n_features} circuit overtaking features")

        return features

    def get_feature_names(self) -> list[str]:
        """Get list of feature names this extractor produces."""
        names = [
            "circuit_avg_positions_changed",
            "circuit_max_positions_changed",
            "circuit_positions_changed_std",
            "circuit_overtaking_rate",
            "circuit_difficulty",
            "grid_importance",
            "circuit_race_count",
            "low_overtaking_circuit",
            "high_overtaking_circuit",
        ]

        for window in self.windows:
            names.extend(
                [
                    f"driver_avg_overtakes_{window}",
                    f"driver_overtakes_std_{window}",
                    f"driver_gain_rate_{window}",
                    f"driver_max_overtakes_{window}",
                    f"overtake_potential_{window}",
                ]
            )

        names.extend(
            [
                "driver_career_avg_overtakes",
                "driver_career_overtake_rate",
                "overtaking_specialist",
                "position_defender",
            ]
        )

        return names


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from src.data.loaders import F1DataLoader

    loader = F1DataLoader()

    # Load race results
    race_results = loader.load_race_results(min_year=2020)
    print(f"Loaded {len(race_results)} race results")

    # Extract features
    extractor = CircuitOvertakingExtractor()
    features = extractor.extract_features(race_results, min_year=2020)

    print(f"\nFeature columns: {features.columns.tolist()}")
    print(f"\nFeature shape: {features.shape}")

    # Check circuit difficulty distribution
    print("\nCircuit difficulty stats:")
    print(features["circuit_difficulty"].describe())

    # Show circuits by difficulty
    circuit_col = "circuit_id" if "circuit_id" in race_results.columns else "circuit"
    if circuit_col in race_results.columns:
        circuit_diff = features.merge(
            race_results[["session_key", circuit_col]].drop_duplicates(),
            on="session_key",
        )
        print("\nCircuit difficulty (higher = harder to overtake):")
        print(
            circuit_diff.groupby(circuit_col)["circuit_difficulty"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )
