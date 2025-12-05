"""
Form momentum feature extraction.

Computes trend-based features including linear regression slopes,
exponentially weighted averages, and acceleration of form.
These features capture improving/declining form trends beyond simple averages.

All features use .shift(1) for temporal safety.
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class MomentumFeatureExtractor:
    """Extracts form momentum and trend features."""

    def __init__(
        self,
        windows: list[int] | None = None,
        ewm_spans: list[int] | None = None,
    ):
        """
        Initialize momentum feature extractor.

        Args:
            windows: Rolling window sizes for trend features (default: [5, 10])
            ewm_spans: Exponential weighted moving average spans (default: [3, 5])
        """
        self.windows = windows or [5, 10]
        self.ewm_spans = ewm_spans or [3, 5]
        logger.info(
            f"Initialized MomentumFeatureExtractor (windows={self.windows}, ewm={self.ewm_spans})"
        )

    def extract_features(
        self,
        qualifying_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Extract momentum features per driver per session.

        Args:
            qualifying_results: DataFrame with qualifying results

        Returns:
            DataFrame with momentum features per driver per session
        """
        logger.info("Extracting momentum features...")

        if qualifying_results.empty:
            logger.warning("No qualifying results provided")
            return pd.DataFrame()

        required_cols = ["session_key", "driver_code", "year", "round", "position"]
        if not all(c in qualifying_results.columns for c in required_cols):
            logger.warning("Missing required columns for momentum features")
            return pd.DataFrame()

        # Start with base DataFrame
        features = qualifying_results[required_cols].copy()

        # Sort for proper time series calculations
        features = features.sort_values(["driver_code", "year", "round"])

        # Add exponentially weighted averages
        features = self._add_ewm_features(features)

        # Add linear trend features
        features = self._add_trend_features(features)

        # Add form acceleration
        features = self._add_acceleration_features(features)

        # Add recent vs long-term comparison
        features = self._add_form_comparison(features)

        # Count features added
        n_features = len([c for c in features.columns if c not in required_cols])
        logger.info(f"Extracted {n_features} momentum features")

        return features

    def _add_ewm_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add exponentially weighted moving average features."""
        for span in self.ewm_spans:
            # EWM position (more recent races weighted higher)
            features[f"ewm_position_{span}"] = features.groupby("driver_code")[
                "position"
            ].transform(lambda x: x.shift(1).ewm(span=span, min_periods=1).mean())

            # EWM top-3 rate
            features["_is_top3"] = (features["position"] <= 3).astype(int)
            features[f"ewm_top3_rate_{span}"] = features.groupby("driver_code")[
                "_is_top3"
            ].transform(lambda x: x.shift(1).ewm(span=span, min_periods=1).mean())

        features = features.drop(columns=["_is_top3"], errors="ignore")
        return features

    def _add_trend_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add linear trend features (position slope over window)."""
        for window in self.windows:
            trends = []
            for driver in features["driver_code"].unique():
                driver_mask = features["driver_code"] == driver
                driver_positions = features.loc[driver_mask, "position"].values

                driver_trends = []
                for i in range(len(driver_positions)):
                    if i < window:
                        # Not enough data for trend
                        driver_trends.append(np.nan)
                    else:
                        # Get previous window positions (shifted, not including current)
                        window_positions = driver_positions[i - window : i]
                        slope = self._compute_slope(window_positions)
                        driver_trends.append(slope)

                trends.extend(driver_trends)

            # Reconstruct in correct order
            driver_order = []
            for driver in features["driver_code"].unique():
                driver_order.extend(features[features["driver_code"] == driver].index.tolist())

            trend_series = pd.Series(trends, index=driver_order)
            features[f"position_trend_{window}"] = trend_series

        return features

    def _compute_slope(self, positions: np.ndarray) -> float:
        """Compute linear regression slope of positions."""
        if len(positions) < 2:
            return np.nan

        x = np.arange(len(positions))
        try:
            slope, _, _, _, _ = stats.linregress(x, positions)
            return slope
        except Exception:
            return np.nan

    def _add_acceleration_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add form acceleration (change in trend)."""
        if len(self.windows) < 2:
            return features

        short_window = min(self.windows)
        long_window = max(self.windows)

        short_trend_col = f"position_trend_{short_window}"
        long_trend_col = f"position_trend_{long_window}"

        if short_trend_col in features.columns and long_trend_col in features.columns:
            # Acceleration: short-term trend minus long-term trend
            # Negative = improving faster recently, Positive = declining faster
            features["form_acceleration"] = features[short_trend_col] - features[long_trend_col]

            # Momentum reversal: short-term contradicts long-term
            features["momentum_reversal"] = (
                features[short_trend_col] * features[long_trend_col] < 0
            ).astype(int)

        return features

    def _add_form_comparison(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add comparison between recent and longer-term form."""
        if len(self.ewm_spans) < 2:
            return features

        short_span = min(self.ewm_spans)
        long_span = max(self.ewm_spans)

        short_ewm_col = f"ewm_position_{short_span}"
        long_ewm_col = f"ewm_position_{long_span}"

        if short_ewm_col in features.columns and long_ewm_col in features.columns:
            # Recent vs long-term form
            # Negative = recent form better than long-term
            features["recent_vs_longterm"] = features[short_ewm_col] - features[long_ewm_col]

            # Form improvement indicator
            features["form_improving"] = (features["recent_vs_longterm"] < 0).astype(int)

        return features

    def get_feature_names(self) -> list[str]:
        """Get list of feature names this extractor produces."""
        names = []

        for span in self.ewm_spans:
            names.extend(
                [
                    f"ewm_position_{span}",
                    f"ewm_top3_rate_{span}",
                ]
            )

        for window in self.windows:
            names.append(f"position_trend_{window}")

        names.extend(
            [
                "form_acceleration",
                "momentum_reversal",
                "recent_vs_longterm",
                "form_improving",
            ]
        )

        return names
