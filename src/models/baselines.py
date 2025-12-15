"""
Baseline prediction models for F1 predictions.

Implements baseline prediction methods:
1. Championship Order: Current top-3 in driver standings
2. Recent Form: Best avg finish in last 3 races
3. Historical Circuit: Previous year's top-3 at same circuit
4. Qualifying-Based (race only): Predict race from grid positions
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class ChampionshipOrderBaseline:
    """Predicts current top-3 in driver championship standings."""

    def __init__(self):
        """Initialize championship order baseline."""
        logger.info("Initialized ChampionshipOrderBaseline")

    def predict(self, standings_df: pd.DataFrame, target_round: int, season_year: int) -> list[str]:
        """
        Predict top-3 based on current championship standings.

        Args:
            standings_df: DataFrame with championship standings
            target_round: Round number for prediction
            season_year: Season year

        Returns:
            List of 3 driver IDs in championship order
        """
        logger.info(
            f"Predicting using championship order for {season_year} round {target_round}..."
        )

        # Get standings from previous round (temporal safety)
        previous_round = target_round - 1
        standings = standings_df[
            (standings_df["season_year"] == season_year)
            & (standings_df["after_round"] == previous_round)
        ].copy()

        if standings.empty:
            logger.warning("No standings data available, using default prediction")
            return ["VER", "HAM", "LEC"]  # Default fallback

        # Sort by position and take top-3
        standings = standings.sort_values("position")
        top_3 = standings.head(3)["driver_id"].tolist()

        logger.info(f"Championship order prediction: {top_3}")
        return top_3


class RecentFormBaseline:
    """Predicts based on best average finish in last N races."""

    def __init__(self, window: int = 3):
        """
        Initialize recent form baseline.

        Args:
            window: Number of recent races to consider (default: 3)
        """
        self.window = window
        logger.info(f"Initialized RecentFormBaseline (window={window})")

    def predict(
        self, race_results_df: pd.DataFrame, races_df: pd.DataFrame, target_date: str
    ) -> list[str]:
        """
        Predict top-3 based on recent form (average position in last N races).

        Args:
            race_results_df: DataFrame with race results
            races_df: DataFrame with race metadata
            target_date: Date of target race (for temporal filtering)

        Returns:
            List of 3 driver IDs with best recent form
        """
        logger.info(f"Predicting using recent form (last {self.window} races)...")

        # Merge with race dates
        df = race_results_df.merge(races_df[["raceId", "date"]], on="raceId", how="left")

        # Filter to races before target date
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] < pd.to_datetime(target_date)]

        if df.empty:
            logger.warning("No historical data available")
            return ["VER", "HAM", "LEC"]

        # Sort by date and get last N races per driver
        df = df.sort_values(["driverId", "date"])

        # Calculate average finish position in last N races
        recent_form = []

        for driver_id in df["driverId"].unique():
            driver_df = df[df["driverId"] == driver_id].tail(self.window)
            avg_position = driver_df["position"].mean()

            if pd.notna(avg_position):
                recent_form.append({"driver_id": driver_id, "avg_position": avg_position})

        # Sort by average position (lower is better) and take top-3
        recent_form_df = pd.DataFrame(recent_form).sort_values("avg_position")
        top_3 = recent_form_df.head(3)["driver_id"].tolist()

        logger.info(f"Recent form prediction: {top_3}")
        return top_3


class HistoricalCircuitBaseline:
    """Predicts based on previous year's results at the same circuit."""

    def __init__(self):
        """Initialize historical circuit baseline."""
        logger.info("Initialized HistoricalCircuitBaseline")

    def predict(
        self,
        race_results_df: pd.DataFrame,
        races_df: pd.DataFrame,
        target_circuit_id: str,
        target_season: int,
    ) -> list[str]:
        """
        Predict top-3 based on previous year's results at same circuit.

        Args:
            race_results_df: DataFrame with race results
            races_df: DataFrame with race metadata
            target_circuit_id: Circuit ID for prediction
            target_season: Season year for prediction

        Returns:
            List of 3 driver IDs from previous year's podium
        """
        logger.info(
            f"Predicting using historical circuit data for {target_circuit_id} ({target_season})..."
        )

        # Find previous year's race at this circuit
        previous_year = target_season - 1
        previous_race = races_df[
            (races_df["circuitId"] == target_circuit_id) & (races_df["year"] == previous_year)
        ]

        if previous_race.empty:
            logger.warning(f"No previous year data for {target_circuit_id}")
            return ["VER", "HAM", "LEC"]

        race_id = previous_race.iloc[0]["raceId"]

        # Get previous year's podium
        podium = race_results_df[race_results_df["raceId"] == race_id].copy()
        podium = podium.sort_values("position")
        top_3 = podium.head(3)["driverId"].tolist()

        logger.info(f"Historical circuit prediction: {top_3}")
        return top_3


class QualifyingBasedBaseline:
    """Predicts race results based on qualifying grid positions (race baseline only)."""

    def __init__(self):
        """Initialize qualifying-based baseline."""
        logger.info("Initialized QualifyingBasedBaseline")

    def predict(self, qualifying_results: list[str]) -> list[str]:
        """
        Predict race top-3 from qualifying results (simple: top-3 quali = top-3 race).

        Args:
            qualifying_results: List of driver IDs in qualifying order

        Returns:
            List of 3 driver IDs (same as qualifying order)
        """
        logger.info("Predicting race results from qualifying positions...")

        if len(qualifying_results) < 3:
            logger.warning("Insufficient qualifying data")
            return ["VER", "HAM", "LEC"]

        top_3 = qualifying_results[:3]
        logger.info(f"Qualifying-based prediction: {top_3}")
        return top_3


class PracticeBasedBaseline:
    """Predicts qualifying results based on FP3 session times."""

    def __init__(self):
        """Initialize practice-based baseline."""
        logger.info("Initialized PracticeBasedBaseline")

    def predict(self, fp3_results: pd.DataFrame) -> list[str]:
        """
        Predict qualifying top-3 from FP3 best lap times.

        Uses FP3 as the most representative practice session for qualifying pace.

        Args:
            fp3_results: DataFrame with columns ['driver_code', 'lap_time_ms']
                        containing FP3 best laps per driver

        Returns:
            List of 3 driver codes with best FP3 times
        """
        logger.info("Predicting qualifying results from FP3 practice times...")

        if fp3_results.empty or len(fp3_results) < 3:
            logger.warning("Insufficient FP3 data")
            return ["VER", "HAM", "LEC"]

        # Sort by lap time (lower is better)
        fp3_sorted = fp3_results.sort_values("lap_time_ms")
        top_3 = fp3_sorted.head(3)["driver_code"].tolist()

        logger.info(f"Practice-based prediction: {top_3}")
        return top_3

    def predict_from_loader(
        self,
        loader,
        year: int,
        round_num: int,
    ) -> list[str]:
        """
        Predict qualifying top-3 using data loader.

        For standard weekends, uses FP3 (most representative practice session).
        For sprint weekends (no FP3), falls back to SQ results then FP1.

        Args:
            loader: F1DataLoader instance
            year: Season year
            round_num: Round number

        Returns:
            List of 3 driver codes with best practice times
        """
        # Try FP3 first (standard weekends)
        fp3_session = loader.load_session(year, round_num, "FP3")

        if not fp3_session.empty:
            fp3_best = fp3_session.groupby("driver_code")["lap_time_ms"].min().reset_index()
            return self.predict(fp3_best)

        # Sprint weekend: FP3 doesn't exist, use SQ results instead
        # (SQ happens before Q on sprint weekends, so SQ results are available)
        if loader.is_sprint_weekend(year, round_num):
            sq_top3 = self._get_sq_top3_for_qualifying_baseline(loader, year, round_num)
            if sq_top3:
                logger.info("Using SQ results as qualifying baseline for sprint weekend")
                return sq_top3

        logger.warning(f"No practice/SQ data for {year} R{round_num}")
        return ["VER", "HAM", "LEC"]

    def _get_sq_top3_for_qualifying_baseline(
        self,
        loader,
        year: int,
        round_num: int,
    ) -> list[str] | None:
        """
        Get SQ top-3 for use as qualifying baseline on sprint weekends.

        On sprint weekends, SQ happens before Q, so SQ results are a valid
        baseline for predicting Q results.

        Args:
            loader: F1DataLoader instance
            year: Season year
            round_num: Round number

        Returns:
            List of 3 driver codes from SQ, or None if unavailable
        """
        # Get SQ positions from Sprint Race grid
        sprint_session = loader.load_session(year, round_num, "S")

        if not sprint_session.empty and "grid_position" in sprint_session.columns:
            grid = sprint_session.groupby("driver_code")["grid_position"].first()
            grid = grid[grid.notna() & (grid > 0)].sort_values()

            if len(grid) >= 3:
                return grid.head(3).index.tolist()

        return None


class FP1BasedBaseline:
    """Predicts sprint qualifying results based on FP1 session times."""

    def __init__(self):
        """Initialize FP1-based baseline."""
        logger.info("Initialized FP1BasedBaseline")

    def predict(self, fp1_results: pd.DataFrame) -> list[str]:
        """
        Predict sprint qualifying top-3 from FP1 best lap times.

        Uses FP1 as the only available practice session before sprint qualifying.

        Args:
            fp1_results: DataFrame with columns ['driver_code', 'lap_time_ms']
                        containing FP1 best laps per driver

        Returns:
            List of 3 driver codes with best FP1 times
        """
        logger.info("Predicting sprint qualifying results from FP1 practice times...")

        if fp1_results.empty or len(fp1_results) < 3:
            logger.warning("Insufficient FP1 data")
            return ["VER", "HAM", "LEC"]

        # Sort by lap time (lower is better)
        fp1_sorted = fp1_results.sort_values("lap_time_ms")
        top_3 = fp1_sorted.head(3)["driver_code"].tolist()

        logger.info(f"FP1-based prediction: {top_3}")
        return top_3

    def predict_from_loader(
        self,
        loader,
        year: int,
        round_num: int,
    ) -> list[str]:
        """
        Predict sprint qualifying top-3 using data loader.

        Args:
            loader: F1DataLoader instance
            year: Season year
            round_num: Round number

        Returns:
            List of 3 driver codes with best FP1 times
        """
        # Load FP1 session
        fp1_session = loader.load_session(year, round_num, "FP1")

        if fp1_session.empty:
            logger.warning(f"No FP1 data for {year} R{round_num}")
            return ["VER", "HAM", "LEC"]

        # Get best lap per driver
        fp1_best = fp1_session.groupby("driver_code")["lap_time_ms"].min().reset_index()

        return self.predict(fp1_best)


class SprintQualiBasedBaseline:
    """Predicts sprint race results based on sprint qualifying grid positions."""

    def __init__(self):
        """Initialize sprint quali-based baseline."""
        logger.info("Initialized SprintQualiBasedBaseline")

    def predict(self, sq_results: list[str]) -> list[str]:
        """
        Predict sprint race top-3 from sprint qualifying results.

        Sprint races are short (~20 laps), so grid position is even more
        predictive than in main races.

        Args:
            sq_results: List of driver codes in sprint qualifying order

        Returns:
            List of 3 driver codes (same as SQ order - grid = finish assumption)
        """
        logger.info("Predicting sprint race results from sprint qualifying positions...")

        if len(sq_results) < 3:
            logger.warning("Insufficient sprint qualifying data")
            return ["VER", "HAM", "LEC"]

        top_3 = sq_results[:3]
        logger.info(f"Sprint quali-based prediction: {top_3}")
        return top_3

    def predict_from_loader(
        self,
        loader,
        year: int,
        round_num: int,
    ) -> list[str]:
        """
        Predict sprint race top-3 using data loader.

        Gets SQ results from Sprint Race grid_position (which is set by SQ).
        This works even when FastF1/Ergast doesn't provide SQ positions directly.

        Args:
            loader: F1DataLoader instance
            year: Season year
            round_num: Round number

        Returns:
            List of 3 driver codes in SQ order
        """
        # Primary method: Get SQ positions from Sprint Race grid
        sprint_session = loader.load_session(year, round_num, "S")

        if not sprint_session.empty and "grid_position" in sprint_session.columns:
            # Sprint Race grid_position equals SQ finishing position
            grid = sprint_session.groupby("driver_code")["grid_position"].first()
            grid = grid[grid.notna() & (grid > 0)].sort_values()

            if len(grid) >= 3:
                top_3 = grid.head(3).index.tolist()
                logger.info(f"SQ baseline from Sprint grid: {top_3}")
                return top_3

        # Fallback: Load SQ results directly (works for 2021-2023)
        sq_results = loader.load_sprint_qualifying_results(min_year=year)
        sq_results = sq_results[
            (sq_results["year"] == year) & (sq_results["round"] == round_num)
        ].copy()

        if not sq_results.empty and sq_results["position"].notna().any():
            sq_results = sq_results[sq_results["position"].notna()].sort_values("position")
            top_3 = sq_results.head(3)["driver_code"].tolist()
            logger.info(f"SQ baseline from SQ results: {top_3}")
            return top_3

        logger.warning(f"No SQ data for {year} R{round_num}")
        return ["VER", "HAM", "LEC"]


class RaceGridBaseline:
    """Predicts race results based on qualifying grid positions."""

    def __init__(self):
        """Initialize race grid baseline."""
        logger.info("Initialized RaceGridBaseline")

    def predict(self, qualifying_results: list[str]) -> list[str]:
        """
        Predict race top-3 from qualifying grid positions.

        Grid position is the most predictive feature for race results.
        This is a simple baseline: top-3 on grid = top-3 in race.

        Args:
            qualifying_results: List of driver codes in qualifying order (grid)

        Returns:
            List of 3 driver codes in grid order
        """
        logger.info("Predicting race results from qualifying grid positions...")

        if len(qualifying_results) < 3:
            logger.warning("Insufficient qualifying data")
            return ["VER", "HAM", "LEC"]

        top_3 = qualifying_results[:3]
        logger.info(f"Grid-based race prediction: {top_3}")
        return top_3

    def predict_from_loader(
        self,
        loader,
        year: int,
        round_num: int,
    ) -> list[str]:
        """
        Predict race top-3 using data loader.

        Args:
            loader: F1DataLoader instance
            year: Season year
            round_num: Round number

        Returns:
            List of 3 driver codes in qualifying order
        """
        # Load qualifying results
        quali_results = loader.load_qualifying_results(min_year=year)
        quali_results = quali_results[
            (quali_results["year"] == year) & (quali_results["round"] == round_num)
        ].copy()

        if quali_results.empty:
            logger.warning(f"No qualifying data for {year} R{round_num}")
            return ["VER", "HAM", "LEC"]

        # Sort by position and get top 3
        quali_results = quali_results.sort_values("position")
        return quali_results.head(3)["driver_code"].tolist()


class BaselineEvaluator:
    """Evaluate all baseline models and compare."""

    def __init__(self):
        """Initialize baseline evaluator."""
        self.championship_baseline = ChampionshipOrderBaseline()
        self.recent_form_baseline = RecentFormBaseline(window=3)
        self.circuit_baseline = HistoricalCircuitBaseline()
        self.qualifying_baseline = QualifyingBasedBaseline()
        self.fp1_baseline = FP1BasedBaseline()
        self.sq_baseline = SprintQualiBasedBaseline()
        self.race_grid_baseline = RaceGridBaseline()
        logger.info("Initialized BaselineEvaluator with all baselines")

    def evaluate_all_baselines(
        self,
        race_results_df: pd.DataFrame,
        races_df: pd.DataFrame,
        standings_df: pd.DataFrame,
        test_season: int = 2024,
    ) -> dict:
        """
        Evaluate all baseline models on test season.

        Args:
            race_results_df: DataFrame with race results
            races_df: DataFrame with race metadata
            standings_df: DataFrame with championship standings
            test_season: Season to use for testing

        Returns:
            Dictionary with baseline scores
        """
        logger.info(f"Evaluating all baselines on {test_season} season...")

        baseline_scores = {
            "Championship Order": [],
            "Recent Form": [],
            "Historical Circuit": [],
        }

        # Get all races in test season
        test_races = races_df[races_df["year"] == test_season]

        for _, race in test_races.iterrows():
            # Get actual podium
            actual_results = race_results_df[race_results_df["raceId"] == race["raceId"]].copy()
            actual_results = actual_results.sort_values("position")
            actual_top3 = actual_results.head(3)["driverId"].tolist()

            if len(actual_top3) < 3:
                continue  # Skip if incomplete data

            # Evaluate championship order
            try:
                champ_pred = self.championship_baseline.predict(
                    standings_df, race["round"], test_season
                )
                champ_score = self._calculate_score(champ_pred, actual_top3)
                baseline_scores["Championship Order"].append(champ_score)
            except Exception as e:
                logger.warning(f"Championship baseline failed: {e}")

            # Evaluate recent form
            try:
                form_pred = self.recent_form_baseline.predict(
                    race_results_df, races_df, race["date"]
                )
                form_score = self._calculate_score(form_pred, actual_top3)
                baseline_scores["Recent Form"].append(form_score)
            except Exception as e:
                logger.warning(f"Recent form baseline failed: {e}")

            # Evaluate historical circuit
            try:
                circuit_pred = self.circuit_baseline.predict(
                    race_results_df, races_df, race["circuitId"], test_season
                )
                circuit_score = self._calculate_score(circuit_pred, actual_top3)
                baseline_scores["Historical Circuit"].append(circuit_score)
            except Exception as e:
                logger.warning(f"Circuit baseline failed: {e}")

        # Calculate average scores
        avg_scores = {
            name: sum(scores) / len(scores) if scores else 0.0
            for name, scores in baseline_scores.items()
        }

        logger.info("Baseline evaluation complete:")
        for name, score in avg_scores.items():
            logger.info(f"  {name}: {score:.2f} avg points/race")

        return avg_scores

    @staticmethod
    def _calculate_score(predicted: list[str], actual: list[str]) -> int:
        """Calculate game points (2 pts exact, 1 pt correct driver)."""
        score = 0
        for i, pred_driver in enumerate(predicted[:3]):
            if pred_driver == actual[i]:
                score += 2
            elif pred_driver in actual:
                score += 1
        return score


def main():
    """Command-line interface for baseline evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Train and evaluate baseline models")
    parser.add_argument("--train", action="store_true", help="Train baseline models")
    parser.add_argument("--season", type=int, default=2024, help="Season to evaluate")
    parser.add_argument(
        "--input", type=str, default="data/processed/", help="Input directory with data"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    BaselineEvaluator()
    logger.info("Baseline evaluation ready")

    # In a real implementation, would load data and run evaluation
    logger.info(f"To evaluate on season {args.season}, load data and call evaluate_all_baselines()")


if __name__ == "__main__":
    main()
