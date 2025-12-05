"""
Qualifying progression feature extraction.

Extracts features based on Q1→Q2→Q3 progression patterns for qualifying prediction.
All features use temporal safety patterns (shift before rolling) to prevent data leakage.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class QualifyingFeatureExtractor:
    """Extracts qualifying progression features from session data."""

    def __init__(self, windows: list[int] | None = None):
        """
        Initialize qualifying feature extractor.

        Args:
            windows: Rolling window sizes for aggregations (default: [3, 5, 10])
        """
        self.windows = windows or [3, 5, 10]

    def extract_features(self, quali_sessions: pd.DataFrame) -> pd.DataFrame:
        """
        Extract qualifying progression features.

        Args:
            quali_sessions: DataFrame with qualifying lap data including quali_session column

        Returns:
            DataFrame with qualifying progression features per driver per session
        """
        if quali_sessions.empty:
            return pd.DataFrame()

        # Compute per-session qualifying stats
        session_features = self._compute_session_quali_stats(quali_sessions)

        if session_features.empty:
            return pd.DataFrame()

        # Compute rolling features across sessions
        rolling_features = self._compute_rolling_quali_features(session_features)

        return rolling_features

    def _compute_session_quali_stats(self, quali_sessions: pd.DataFrame) -> pd.DataFrame:
        """Compute qualifying statistics for each driver per session."""
        required_cols = ["session_key", "driver_code", "position"]
        missing = [c for c in required_cols if c not in quali_sessions.columns]
        if missing:
            logger.warning(f"Missing columns for qualifying features: {missing}")
            return pd.DataFrame()

        # Get Q times if available
        has_q_times = all(
            c in quali_sessions.columns for c in ["q1_time_ms", "q2_time_ms", "q3_time_ms"]
        )

        # Get quali_session split if available
        has_quali_session = "quali_session" in quali_sessions.columns

        # First aggregate by driver per session
        grouped = quali_sessions.groupby(["session_key", "driver_code"])

        agg_dict = {
            "year": "first",
            "round": "first",
            "circuit": "first",
            "team": "first",
            "position": "first",  # Final qualifying position
        }

        if has_q_times:
            agg_dict.update(
                {
                    "q1_time_ms": "first",
                    "q2_time_ms": "first",
                    "q3_time_ms": "first",
                }
            )

        session_stats = grouped.agg(agg_dict).reset_index()

        # Calculate Q1/Q2/Q3 progression
        if has_q_times:
            # Did driver make Q2? (Q2 time exists)
            session_stats["made_q2"] = session_stats["q2_time_ms"].notna()
            # Did driver make Q3? (Q3 time exists)
            session_stats["made_q3"] = session_stats["q3_time_ms"].notna()

            # Q1→Q2 improvement (negative = faster)
            session_stats["q1_to_q2_delta_ms"] = (
                session_stats["q2_time_ms"] - session_stats["q1_time_ms"]
            )

            # Q2→Q3 improvement (negative = faster)
            session_stats["q2_to_q3_delta_ms"] = (
                session_stats["q3_time_ms"] - session_stats["q2_time_ms"]
            )

            # Q1→Q3 total improvement
            session_stats["q1_to_q3_delta_ms"] = (
                session_stats["q3_time_ms"] - session_stats["q1_time_ms"]
            )

            # Improvement percentages
            session_stats["q1_to_q2_pct"] = (
                session_stats["q1_to_q2_delta_ms"] / session_stats["q1_time_ms"] * 100
            )
            session_stats["q2_to_q3_pct"] = (
                session_stats["q2_to_q3_delta_ms"] / session_stats["q2_time_ms"] * 100
            )

        # Count laps per quali session if available
        if has_quali_session:
            q_laps = (
                quali_sessions.groupby(["session_key", "driver_code", "quali_session"])
                .size()
                .unstack(fill_value=0)
            )

            if "Q1" in q_laps.columns:
                session_stats = session_stats.merge(
                    q_laps["Q1"].reset_index().rename(columns={"Q1": "q1_laps"}),
                    on=["session_key", "driver_code"],
                    how="left",
                )
            if "Q2" in q_laps.columns:
                session_stats = session_stats.merge(
                    q_laps["Q2"].reset_index().rename(columns={"Q2": "q2_laps"}),
                    on=["session_key", "driver_code"],
                    how="left",
                )
            if "Q3" in q_laps.columns:
                session_stats = session_stats.merge(
                    q_laps["Q3"].reset_index().rename(columns={"Q3": "q3_laps"}),
                    on=["session_key", "driver_code"],
                    how="left",
                )

        # Position-based features
        session_stats["in_top_3"] = session_stats["position"] <= 3
        session_stats["in_top_10"] = session_stats["position"] <= 10
        session_stats["on_pole"] = session_stats["position"] == 1
        session_stats["front_row"] = session_stats["position"] <= 2

        # Calculate elimination margin (gap to P15/P10 cutoff)
        self._add_elimination_margins(session_stats)

        return session_stats

    def _add_elimination_margins(self, df: pd.DataFrame) -> None:
        """Add Q1/Q2 elimination margin features."""
        if "q1_time_ms" not in df.columns:
            return

        # For each session, calculate the cutoff times
        for session_key in df["session_key"].unique():
            mask = df["session_key"] == session_key

            # Q1 elimination cutoff (P15 time - top 15 advance to Q2)
            q1_times = df.loc[mask, "q1_time_ms"].dropna().sort_values()
            if len(q1_times) >= 15:
                q1_cutoff = q1_times.iloc[14]  # P15 time
                df.loc[mask, "q1_margin_to_cutoff_ms"] = q1_cutoff - df.loc[mask, "q1_time_ms"]

            # Q2 elimination cutoff (P10 time - top 10 advance to Q3)
            q2_times = df.loc[mask, "q2_time_ms"].dropna().sort_values()
            if len(q2_times) >= 10:
                q2_cutoff = q2_times.iloc[9]  # P10 time
                df.loc[mask, "q2_margin_to_cutoff_ms"] = q2_cutoff - df.loc[mask, "q2_time_ms"]

    def _compute_rolling_quali_features(self, session_stats: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling features across sessions with temporal shift."""
        if session_stats.empty:
            return pd.DataFrame()

        # Sort by driver and session
        df = session_stats.sort_values(["driver_code", "year", "round"]).copy()

        features = df[
            ["session_key", "driver_code", "year", "round", "circuit", "team", "position"]
        ].copy()

        # Rolling aggregations for each window
        for window in self.windows:
            # Q2/Q3 advancement rates
            if "made_q2" in df.columns:
                features[f"q2_rate_{window}"] = df.groupby("driver_code")["made_q2"].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                )

            if "made_q3" in df.columns:
                features[f"q3_rate_{window}"] = df.groupby("driver_code")["made_q3"].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                )

            # Position rates
            features[f"top3_rate_{window}"] = df.groupby("driver_code")["in_top_3"].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

            features[f"top10_rate_{window}"] = df.groupby("driver_code")["in_top_10"].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

            features[f"pole_rate_{window}"] = df.groupby("driver_code")["on_pole"].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

            features[f"front_row_rate_{window}"] = df.groupby("driver_code")["front_row"].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

            # Average position
            features[f"avg_position_{window}"] = df.groupby("driver_code")["position"].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

            # Position consistency (std)
            features[f"position_std_{window}"] = df.groupby("driver_code")["position"].transform(
                lambda x: x.shift(1).rolling(window, min_periods=2).std()
            )

            # Q progression improvement rates
            if "q1_to_q2_pct" in df.columns:
                features[f"avg_q1_to_q2_improvement_{window}"] = df.groupby("driver_code")[
                    "q1_to_q2_pct"
                ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

            if "q2_to_q3_pct" in df.columns:
                features[f"avg_q2_to_q3_improvement_{window}"] = df.groupby("driver_code")[
                    "q2_to_q3_pct"
                ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

            # Elimination margin averages
            if "q1_margin_to_cutoff_ms" in df.columns:
                features[f"avg_q1_margin_{window}"] = df.groupby("driver_code")[
                    "q1_margin_to_cutoff_ms"
                ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

            if "q2_margin_to_cutoff_ms" in df.columns:
                features[f"avg_q2_margin_{window}"] = df.groupby("driver_code")[
                    "q2_margin_to_cutoff_ms"
                ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

        # Final run specialist (improves in final laps of session)
        features["final_run_specialist"] = self._compute_final_run_specialist(df)

        return features

    def _compute_final_run_specialist(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute whether driver typically improves on final run.

        Returns float representing rate of improvement on final Q3 run.
        """
        # This would require lap-by-lap timing which we may not have aggregated
        # For now, use Q2→Q3 improvement as proxy
        if "q2_to_q3_pct" not in df.columns:
            return pd.Series(0.0, index=df.index)

        # Positive improvement rate = typically improves from Q2 to Q3
        result = df.groupby("driver_code")["q2_to_q3_pct"].transform(
            lambda x: (x.shift(1) < 0).rolling(10, min_periods=1).mean()
        )

        return result

    def extract_team_qualifying_features(self, quali_sessions: pd.DataFrame) -> pd.DataFrame:
        """
        Extract team-level qualifying features.

        Args:
            quali_sessions: DataFrame with qualifying data

        Returns:
            DataFrame with team qualifying features
        """
        session_stats = self._compute_session_quali_stats(quali_sessions)

        if session_stats.empty:
            return pd.DataFrame()

        # Group by session and team
        team_stats = (
            session_stats.groupby(["session_key", "team"])
            .agg(
                {
                    "year": "first",
                    "round": "first",
                    "position": ["min", "mean"],  # Best and average position
                    "made_q3": "mean" if "made_q3" in session_stats.columns else "first",
                }
            )
            .reset_index()
        )

        # Flatten column names
        team_stats.columns = [
            "_".join(col).strip("_") if isinstance(col, tuple) else col
            for col in team_stats.columns
        ]

        return team_stats
