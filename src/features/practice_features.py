"""
Practice session feature extraction.

Extracts features from FP1/FP2/FP3 sessions for qualifying prediction.
These features help predict quali performance based on practice pace.
All features use temporal safety patterns to prevent data leakage.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class PracticeFeatureExtractor:
    """Extracts practice session features for qualifying prediction."""

    def __init__(self, windows: list[int] | None = None):
        """
        Initialize practice feature extractor.

        Args:
            windows: Rolling window sizes for aggregations (default: [3, 5, 10])
        """
        self.windows = windows or [3, 5, 10]

    def extract_features(
        self,
        practice_sessions: pd.DataFrame,
        quali_sessions: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Extract practice-to-qualifying correlation features.

        Args:
            practice_sessions: DataFrame with FP1/FP2/FP3 lap data
            quali_sessions: DataFrame with qualifying lap data (for correlation)

        Returns:
            DataFrame with practice features per driver per race weekend
        """
        if practice_sessions.empty:
            return pd.DataFrame()

        # Compute practice stats per race weekend
        practice_stats = self._compute_weekend_practice_stats(practice_sessions)

        if practice_stats.empty:
            return pd.DataFrame()

        # Compute practice-to-quali correlation if quali data available
        if not quali_sessions.empty:
            practice_stats = self._add_practice_quali_correlation(practice_stats, quali_sessions)

        # Compute rolling features
        rolling_features = self._compute_rolling_practice_features(practice_stats)

        return rolling_features

    def _compute_weekend_practice_stats(self, practice_sessions: pd.DataFrame) -> pd.DataFrame:
        """Compute practice statistics per driver per race weekend."""
        required_cols = ["year", "round", "driver_code", "session_type", "lap_time_ms"]
        missing = [c for c in required_cols if c not in practice_sessions.columns]
        if missing:
            logger.warning(f"Missing columns for practice features: {missing}")
            return pd.DataFrame()

        # Create weekend key
        practice_sessions = practice_sessions.copy()
        practice_sessions["weekend_key"] = (
            practice_sessions["year"].astype(str)
            + "_"
            + practice_sessions["round"].astype(str).str.zfill(2)
        )

        # Filter valid laps
        valid_laps = practice_sessions[
            practice_sessions["lap_time_ms"].notna() & ~practice_sessions.get("deleted", False)
        ].copy()

        if valid_laps.empty:
            return pd.DataFrame()

        # Aggregate by weekend and driver
        weekend_stats = []

        for (weekend, driver), group in valid_laps.groupby(["weekend_key", "driver_code"]):
            year = group["year"].iloc[0]
            round_num = group["round"].iloc[0]
            circuit = group["circuit"].iloc[0] if "circuit" in group.columns else ""
            team = group["team"].iloc[0] if "team" in group.columns else ""

            stats = {
                "weekend_key": weekend,
                "driver_code": driver,
                "year": year,
                "round": round_num,
                "circuit": circuit,
                "team": team,
            }

            # Best lap per session
            # Note: session_type in data is "Practice 1", "Practice 2", "Practice 3"
            session_mapping = {
                "Practice 1": "fp1",
                "Practice 2": "fp2",
                "Practice 3": "fp3",
            }
            for session_type, prefix in session_mapping.items():
                session_laps = group[group["session_type"] == session_type]
                if not session_laps.empty:
                    stats[f"{prefix}_best_ms"] = session_laps["lap_time_ms"].min()
                    stats[f"{prefix}_laps"] = len(session_laps)
                else:
                    stats[f"{prefix}_best_ms"] = None
                    stats[f"{prefix}_laps"] = 0

            # Overall best practice lap
            stats["practice_best_ms"] = group["lap_time_ms"].min()
            stats["total_practice_laps"] = len(group)

            # Practice progression (FP1 → FP3 improvement)
            if stats.get("fp1_best_ms") and stats.get("fp3_best_ms"):
                stats["fp1_to_fp3_improvement_ms"] = stats["fp1_best_ms"] - stats["fp3_best_ms"]
                stats["fp1_to_fp3_improvement_pct"] = (
                    stats["fp1_to_fp3_improvement_ms"] / stats["fp1_best_ms"] * 100
                )

            # Long run pace (laps 5-15 of a stint for race simulation)
            # Approximate by looking at later laps in FP2/FP3 (often long runs)
            if "stint" in group.columns and "tyre_life" in group.columns:
                long_run_laps = group[(group["tyre_life"] >= 5) & (group["tyre_life"] <= 15)]
                if len(long_run_laps) >= 3:
                    stats["long_run_pace_ms"] = long_run_laps["lap_time_ms"].median()

            # Sector times from practice
            if all(
                c in group.columns
                for c in ["sector1_time_ms", "sector2_time_ms", "sector3_time_ms"]
            ):
                stats["practice_best_s1_ms"] = group["sector1_time_ms"].min()
                stats["practice_best_s2_ms"] = group["sector2_time_ms"].min()
                stats["practice_best_s3_ms"] = group["sector3_time_ms"].min()

            weekend_stats.append(stats)

        return pd.DataFrame(weekend_stats)

    def _add_practice_quali_correlation(
        self, practice_stats: pd.DataFrame, quali_sessions: pd.DataFrame
    ) -> pd.DataFrame:
        """Add practice-to-qualifying performance comparison."""
        # Get qualifying best laps per weekend
        quali_best = (
            quali_sessions.groupby(["year", "round", "driver_code"])
            .agg(
                {
                    "lap_time_ms": "min",
                    "position": "first",
                }
            )
            .reset_index()
        )
        quali_best = quali_best.rename(
            columns={
                "lap_time_ms": "quali_best_ms",
                "position": "quali_position",
            }
        )

        # Merge with practice stats
        practice_stats = practice_stats.merge(
            quali_best,
            on=["year", "round", "driver_code"],
            how="left",
        )

        # Calculate practice-to-quali gap
        if "quali_best_ms" in practice_stats.columns:
            # How much faster was quali vs practice?
            practice_stats["practice_to_quali_gap_ms"] = (
                practice_stats["practice_best_ms"] - practice_stats["quali_best_ms"]
            )

            # FP3 to quali gap (most relevant)
            practice_stats["fp3_to_quali_gap_ms"] = (
                practice_stats["fp3_best_ms"] - practice_stats["quali_best_ms"]
            )

        return practice_stats

    def _compute_rolling_practice_features(self, practice_stats: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling practice features with temporal shift."""
        if practice_stats.empty:
            return pd.DataFrame()

        df = practice_stats.sort_values(["driver_code", "year", "round"]).copy()

        features = df[["weekend_key", "driver_code", "year", "round", "circuit", "team"]].copy()

        # Add current weekend practice data (this is from same weekend, OK to use)
        for col in [
            "fp1_best_ms",
            "fp2_best_ms",
            "fp3_best_ms",
            "practice_best_ms",
            "total_practice_laps",
            "fp1_to_fp3_improvement_pct",
        ]:
            if col in df.columns:
                features[f"current_{col}"] = df[col]

        # Add FP rankings within each weekend (relative to field)
        features = self._add_fp_rankings(features, df)

        # Rolling features (from PREVIOUS weekends only - shift(1))
        for window in self.windows:
            # Practice-to-quali correlation (historical)
            if "practice_to_quali_gap_ms" in df.columns:
                features[f"avg_practice_to_quali_gap_{window}"] = df.groupby("driver_code")[
                    "practice_to_quali_gap_ms"
                ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

            if "fp3_to_quali_gap_ms" in df.columns:
                features[f"avg_fp3_to_quali_gap_{window}"] = df.groupby("driver_code")[
                    "fp3_to_quali_gap_ms"
                ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

            # FP improvement rate
            if "fp1_to_fp3_improvement_pct" in df.columns:
                features[f"avg_fp_improvement_{window}"] = df.groupby("driver_code")[
                    "fp1_to_fp3_improvement_pct"
                ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

            # Long run pace (if available)
            if "long_run_pace_ms" in df.columns:
                features[f"avg_long_run_pace_{window}"] = df.groupby("driver_code")[
                    "long_run_pace_ms"
                ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

        return features

    def _add_fp_rankings(
        self, features: pd.DataFrame, practice_stats: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add FP session rankings relative to field.

        These are current weekend features (not historical) - highly predictive
        since FP3 order often correlates with qualifying order.
        """
        import numpy as np

        # FP3 is most predictive of qualifying
        for session, col in [("fp3", "fp3_best_ms"), ("fp2", "fp2_best_ms")]:
            if col not in practice_stats.columns:
                continue

            # Rank within each weekend (lower time = better rank)
            features[f"current_{session}_rank"] = practice_stats.groupby("weekend_key")[col].rank(
                method="min", na_option="bottom"
            )

            # Gap to session leader (in ms)
            session_fastest = practice_stats.groupby("weekend_key")[col].transform("min")
            features[f"current_{session}_gap_ms"] = practice_stats[col] - session_fastest

            # Gap as percentage of fastest time
            features[f"current_{session}_gap_pct"] = (
                (practice_stats[col] - session_fastest) / session_fastest * 100
            ).replace([np.inf, -np.inf], np.nan)

        # Combined practice rank (best of FP1/FP2/FP3)
        if "practice_best_ms" in practice_stats.columns:
            features["current_practice_rank"] = practice_stats.groupby("weekend_key")[
                "practice_best_ms"
            ].rank(method="min", na_option="bottom")

            # Gap to weekend leader
            weekend_fastest = practice_stats.groupby("weekend_key")["practice_best_ms"].transform(
                "min"
            )
            features["current_practice_gap_ms"] = (
                practice_stats["practice_best_ms"] - weekend_fastest
            )
            features["current_practice_gap_pct"] = (
                (practice_stats["practice_best_ms"] - weekend_fastest) / weekend_fastest * 100
            ).replace([np.inf, -np.inf], np.nan)

        return features

    def extract_fp3_rankings(self, practice_sessions: pd.DataFrame) -> pd.DataFrame:
        """
        Extract FP3 rankings as a baseline for qualifying prediction.

        This is useful for the PracticeBasedBaseline model.

        Args:
            practice_sessions: DataFrame with practice lap data

        Returns:
            DataFrame with FP3 rankings per driver per weekend
        """
        if practice_sessions.empty:
            return pd.DataFrame()

        # Filter to FP3 only (session_type is "Practice 3" in the data)
        fp3 = practice_sessions[practice_sessions["session_type"] == "Practice 3"].copy()

        if fp3.empty:
            return pd.DataFrame()

        # Get best lap per driver in FP3
        fp3_best = (
            fp3.groupby(["year", "round", "driver_code"])
            .agg(
                {
                    "lap_time_ms": "min",
                    "circuit": "first",
                    "team": "first",
                }
            )
            .reset_index()
        )

        # Rank within each session
        fp3_best["fp3_rank"] = fp3_best.groupby(["year", "round"])["lap_time_ms"].rank()

        return fp3_best

    def extract_circuit_practice_features(self, practice_sessions: pd.DataFrame) -> pd.DataFrame:
        """
        Extract circuit-specific practice performance features.

        Args:
            practice_sessions: DataFrame with practice lap data

        Returns:
            DataFrame with circuit-specific practice features
        """
        weekend_stats = self._compute_weekend_practice_stats(practice_sessions)

        if weekend_stats.empty:
            return pd.DataFrame()

        # Group by driver and circuit
        circuit_features = (
            weekend_stats.groupby(["driver_code", "circuit"])
            .agg(
                {
                    "practice_best_ms": "mean",
                    "total_practice_laps": "mean",
                    "fp1_to_fp3_improvement_pct": "mean",
                    "weekend_key": "count",
                }
            )
            .reset_index()
        )

        circuit_features = circuit_features.rename(
            columns={
                "weekend_key": "circuit_practice_appearances",
                "practice_best_ms": "circuit_avg_practice_pace",
                "total_practice_laps": "circuit_avg_practice_laps",
                "fp1_to_fp3_improvement_pct": "circuit_avg_improvement",
            }
        )

        return circuit_features
