"""
ELO rating feature extraction for F1 predictions.

Implements driver and constructor ELO ratings inspired by chess ELO system.
Key insight from Kaggle/research: ELO captures driver skill separate from car
performance by comparing drivers against their teammates.

Features:
- Driver ELO: Based on head-to-head vs teammate (isolates driver skill)
- Constructor ELO: Based on team's overall field performance
- Grid-adjusted ELO (gaELO): Adjusts for starting position

References:
- https://matthewperron.github.io/f1-elo/
- https://github.com/mwtmurphy/f1-elo
- Medium: "Predicting Formula 1 results with Elo Ratings"
"""

import logging
from collections import defaultdict

import pandas as pd

logger = logging.getLogger(__name__)

# ELO system constants
INITIAL_ELO = 1500
K_FACTOR_DRIVER = 32  # Standard K-factor for driver comparisons
K_FACTOR_CONSTRUCTOR = 24  # Slightly lower for team stability
ELO_SCALE = 400  # Standard chess scale


class ELORatingSystem:
    """
    ELO rating system for F1 drivers and constructors.

    Uses teammate comparisons for driver ELO (isolates skill from car).
    Uses field comparisons for constructor ELO (measures car performance).
    """

    def __init__(
        self,
        k_driver: float = K_FACTOR_DRIVER,
        k_constructor: float = K_FACTOR_CONSTRUCTOR,
        initial_elo: float = INITIAL_ELO,
    ):
        """
        Initialize ELO rating system.

        Args:
            k_driver: K-factor for driver rating updates
            k_constructor: K-factor for constructor rating updates
            initial_elo: Starting ELO for new drivers/constructors
        """
        self.k_driver = k_driver
        self.k_constructor = k_constructor
        self.initial_elo = initial_elo

        # Rating storage
        self.driver_elos: dict[str, float] = defaultdict(lambda: initial_elo)
        self.constructor_elos: dict[str, float] = defaultdict(lambda: initial_elo)

        logger.info(
            f"Initialized ELORatingSystem (k_driver={k_driver}, k_constructor={k_constructor})"
        )

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Calculate expected score for player A vs player B.

        Returns probability that A beats B (0 to 1).
        """
        return 1 / (1 + 10 ** ((rating_b - rating_a) / ELO_SCALE))

    def update_rating(
        self, current_rating: float, expected: float, actual: float, k: float
    ) -> float:
        """
        Update ELO rating based on result.

        Args:
            current_rating: Current ELO rating
            expected: Expected score (0 to 1)
            actual: Actual score (1=win, 0.5=draw, 0=loss)
            k: K-factor for update magnitude

        Returns:
            New ELO rating
        """
        return current_rating + k * (actual - expected)

    def process_session_driver_elo(
        self,
        results: pd.DataFrame,
    ) -> dict[str, float]:
        """
        Update driver ELO based on teammate comparison for one session.

        Uses only head-to-head teammate comparisons to isolate driver skill
        from car performance.

        Args:
            results: Session results with driver_code, team, position

        Returns:
            Dict of driver ELO changes (for logging)
        """
        changes = {}

        # Group by team to find teammates
        for _team, team_results in results.groupby("team"):
            if len(team_results) < 2:
                continue  # Need 2 drivers to compare

            # Get the two teammates
            teammates = team_results.sort_values("position")
            driver1 = teammates.iloc[0]  # Better finishing position
            driver2 = teammates.iloc[1]  # Worse finishing position

            d1_code = driver1["driver_code"]
            d2_code = driver2["driver_code"]

            # Skip if either DNF (position > 20 or NaN)
            if pd.isna(driver1["position"]) or pd.isna(driver2["position"]):
                continue
            if driver1["position"] > 20 or driver2["position"] > 20:
                continue

            # Get current ratings
            elo1 = self.driver_elos[d1_code]
            elo2 = self.driver_elos[d2_code]

            # Calculate expected scores
            exp1 = self.expected_score(elo1, elo2)
            exp2 = self.expected_score(elo2, elo1)

            # Actual scores: winner gets 1, loser gets 0
            # (could use margin-based scoring but simple is robust)
            actual1 = 1.0
            actual2 = 0.0

            # Update ratings
            new_elo1 = self.update_rating(elo1, exp1, actual1, self.k_driver)
            new_elo2 = self.update_rating(elo2, exp2, actual2, self.k_driver)

            self.driver_elos[d1_code] = new_elo1
            self.driver_elos[d2_code] = new_elo2

            changes[d1_code] = new_elo1 - elo1
            changes[d2_code] = new_elo2 - elo2

        return changes

    def process_session_constructor_elo(
        self,
        results: pd.DataFrame,
    ) -> dict[str, float]:
        """
        Update constructor ELO based on overall field performance.

        Compares each team's best result against all other teams.

        Args:
            results: Session results with team, position

        Returns:
            Dict of constructor ELO changes
        """
        changes = {}

        # Get best position per team
        team_best = results.groupby("team")["position"].min()
        team_best = team_best.dropna()

        if len(team_best) < 2:
            return changes

        teams = team_best.index.tolist()
        positions = team_best.values

        # Pairwise comparisons (round-robin style)
        for i, team1 in enumerate(teams):
            pos1 = positions[i]
            if pd.isna(pos1) or pos1 > 20:
                continue

            elo1 = self.constructor_elos[team1]
            total_expected = 0
            total_actual = 0
            n_comparisons = 0

            for j, team2 in enumerate(teams):
                if i == j:
                    continue

                pos2 = positions[j]
                if pd.isna(pos2) or pos2 > 20:
                    continue

                elo2 = self.constructor_elos[team2]
                exp = self.expected_score(elo1, elo2)
                actual = 1.0 if pos1 < pos2 else (0.5 if pos1 == pos2 else 0.0)

                total_expected += exp
                total_actual += actual
                n_comparisons += 1

            if n_comparisons > 0:
                # Normalize by number of comparisons
                avg_expected = total_expected / n_comparisons
                avg_actual = total_actual / n_comparisons

                new_elo = self.update_rating(elo1, avg_expected, avg_actual, self.k_constructor)
                old_elo = self.constructor_elos[team1]
                self.constructor_elos[team1] = new_elo
                changes[team1] = new_elo - old_elo

        return changes

    def get_driver_elo(self, driver_code: str) -> float:
        """Get current driver ELO rating."""
        return self.driver_elos[driver_code]

    def get_constructor_elo(self, team: str) -> float:
        """Get current constructor ELO rating."""
        return self.constructor_elos[team]


class ELOFeatureExtractor:
    """
    Extract ELO-based features for F1 prediction.

    Features are calculated using temporal shift to prevent data leakage:
    ELO at race N uses only results from races 1 to N-1.
    """

    def __init__(
        self,
        k_driver: float = K_FACTOR_DRIVER,
        k_constructor: float = K_FACTOR_CONSTRUCTOR,
    ):
        """
        Initialize ELO feature extractor.

        Args:
            k_driver: K-factor for driver ELO updates
            k_constructor: K-factor for constructor ELO updates
        """
        self.k_driver = k_driver
        self.k_constructor = k_constructor
        logger.info("Initialized ELOFeatureExtractor")

    def extract_features(
        self,
        results: pd.DataFrame,
        session_type: str = "Q",
    ) -> pd.DataFrame:
        """
        Extract ELO features for all sessions.

        Args:
            results: Historical session results with session_key, driver_code,
                     team, position, year, round
            session_type: Session type for ELO calculation ("Q" or "R")

        Returns:
            DataFrame with ELO features per driver per session
        """
        logger.info(f"Extracting ELO features for {session_type} sessions...")

        if results.empty:
            logger.warning("No results provided")
            return pd.DataFrame()

        required_cols = ["session_key", "driver_code", "team", "position", "year", "round"]
        if not all(c in results.columns for c in required_cols):
            missing = [c for c in required_cols if c not in results.columns]
            logger.warning(f"Missing required columns: {missing}")
            return pd.DataFrame()

        # Initialize ELO system
        elo_system = ELORatingSystem(
            k_driver=self.k_driver,
            k_constructor=self.k_constructor,
        )

        # Sort chronologically
        df = results.copy()
        df = df.sort_values(["year", "round"])

        # Store features for each driver/session BEFORE updating ELO
        feature_records = []

        # Process sessions in order
        for session_key in df["session_key"].unique():
            session_results = df[df["session_key"] == session_key]

            # Record CURRENT ELO (before this session) as features
            for _, row in session_results.iterrows():
                driver_elo = elo_system.get_driver_elo(row["driver_code"])
                constructor_elo = elo_system.get_constructor_elo(row["team"])

                # Grid-adjusted ELO (if grid position available)
                grid_pos = row.get("grid_position", row["position"])
                if pd.notna(grid_pos) and grid_pos > 0:
                    # Adjust ELO based on grid: front grid = boost, back grid = penalty
                    # Scale: P1 gets +50 ELO adjustment, P20 gets -50
                    grid_adjustment = (10.5 - grid_pos) * 5  # P1=+47.5, P10=+2.5, P20=-47.5
                    ga_driver_elo = driver_elo + grid_adjustment
                else:
                    ga_driver_elo = driver_elo

                feature_records.append(
                    {
                        "session_key": session_key,
                        "driver_code": row["driver_code"],
                        "year": row["year"],
                        "round": row["round"],
                        "driver_elo": driver_elo,
                        "constructor_elo": constructor_elo,
                        "ga_driver_elo": ga_driver_elo,
                        # Relative features
                        "driver_elo_vs_initial": driver_elo - INITIAL_ELO,
                        "constructor_elo_vs_initial": constructor_elo - INITIAL_ELO,
                    }
                )

            # NOW update ELO with this session's results
            elo_system.process_session_driver_elo(session_results)
            elo_system.process_session_constructor_elo(session_results)

        features = pd.DataFrame(feature_records)

        # Add relative ELO features (driver vs field average, constructor rank)
        features = self._add_relative_elo_features(features, elo_system)

        n_features = len(
            [
                c
                for c in features.columns
                if c not in ["session_key", "driver_code", "year", "round"]
            ]
        )
        logger.info(f"Extracted {n_features} ELO features for {len(features)} samples")

        return features

    def _add_relative_elo_features(
        self,
        features: pd.DataFrame,
        elo_system: ELORatingSystem,
    ) -> pd.DataFrame:
        """Add relative ELO features comparing to field."""
        if features.empty:
            return features

        # Driver ELO percentile within each session
        features["driver_elo_percentile"] = features.groupby("session_key")["driver_elo"].transform(
            lambda x: x.rank(pct=True)
        )

        # Constructor ELO percentile
        features["constructor_elo_percentile"] = features.groupby("session_key")[
            "constructor_elo"
        ].transform(lambda x: x.rank(pct=True))

        # Driver ELO vs session average
        features["driver_elo_vs_avg"] = features.groupby("session_key")["driver_elo"].transform(
            lambda x: x - x.mean()
        )

        # Constructor ELO vs session average
        features["constructor_elo_vs_avg"] = features.groupby("session_key")[
            "constructor_elo"
        ].transform(lambda x: x - x.mean())

        # Combined ELO (30% driver + 70% constructor, inspired by research)
        # Rationale: Car matters more than driver skill historically
        features["combined_elo"] = 0.3 * features["driver_elo"] + 0.7 * features["constructor_elo"]

        return features

    def get_feature_names(self) -> list[str]:
        """Get list of feature names this extractor produces."""
        return [
            "driver_elo",
            "constructor_elo",
            "ga_driver_elo",
            "driver_elo_vs_initial",
            "constructor_elo_vs_initial",
            "driver_elo_percentile",
            "constructor_elo_percentile",
            "driver_elo_vs_avg",
            "constructor_elo_vs_avg",
            "combined_elo",
        ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from src.data.loaders import F1DataLoader

    loader = F1DataLoader()

    # Load qualifying results
    quali_results = loader.load_qualifying_results(min_year=2022)
    print(f"Loaded {len(quali_results)} qualifying results")

    # Extract ELO features
    extractor = ELOFeatureExtractor()
    features = extractor.extract_features(quali_results, session_type="Q")

    print(f"\nFeature columns: {features.columns.tolist()}")
    print(f"\nFeature shape: {features.shape}")
    print("\nSample features:")
    print(features.head(20))

    # Check ELO distribution
    print("\nDriver ELO stats:")
    print(features["driver_elo"].describe())
    print("\nConstructor ELO stats:")
    print(features["constructor_elo"].describe())
