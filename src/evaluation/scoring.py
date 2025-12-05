"""
Game points scoring function for F1 predictions.

Scoring: 2 points for exact position match, 1 point for correct driver wrong position.
"""

import logging

import numpy as np
from sklearn.metrics import make_scorer

logger = logging.getLogger(__name__)


def calculate_game_points(predicted: list[str], actual: list[str]) -> int:
    """
    Calculate game points for a prediction using family game rules.

    Scoring:
    - 2 points: Correct driver in correct position
    - 1 point: Correct driver in wrong position
    - 0 points: Driver not in top-3

    Args:
        predicted: List of 3 driver IDs in predicted order [P1, P2, P3]
        actual: List of 3 driver IDs in actual order [P1, P2, P3]

    Returns:
        Total points (0-6)

    Examples:
        >>> calculate_game_points(["VER", "LEC", "HAM"], ["VER", "LEC", "HAM"])
        6  # All exact matches
        >>> calculate_game_points(["VER", "HAM", "LEC"], ["VER", "LEC", "HAM"])
        4  # VER exact (2), HAM in top-3 (1), LEC in top-3 (1)
        >>> calculate_game_points(["VER", "NOR", "PIA"], ["HAM", "LEC", "SAI"])
        0  # No matches
    """
    if len(predicted) != 3 or len(actual) != 3:
        raise ValueError("Both predicted and actual must contain exactly 3 drivers")

    score = 0

    for i, pred_driver in enumerate(predicted):
        if pred_driver == actual[i]:
            # Exact position match
            score += 2
            logger.debug(f"Exact match: {pred_driver} at position {i + 1} (+2 points)")
        elif pred_driver in actual:
            # Correct driver, wrong position
            score += 1
            logger.debug(f"Driver match: {pred_driver} in top-3 but wrong position (+1 point)")

    logger.info(f"Prediction scored {score}/6 points")
    return score


def calculate_game_points_array(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate average game points for multiple predictions (for scikit-learn compatibility).

    Args:
        y_true: Array of actual results, shape (n_samples, 3)
        y_pred: Array of predicted results, shape (n_samples, 3)

    Returns:
        Average game points across all predictions
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if y_true.shape[1] != 3:
        raise ValueError("Predictions must be for top-3 positions")

    total_points = 0
    n_predictions = len(y_true)

    for i in range(n_predictions):
        predicted = y_pred[i].tolist()
        actual = y_true[i].tolist()
        points = calculate_game_points(predicted, actual)
        total_points += points

    avg_points = total_points / n_predictions if n_predictions > 0 else 0.0
    logger.info(f"Average game points: {avg_points:.2f} across {n_predictions} predictions")

    return avg_points


def make_game_points_scorer():
    """
    Create a scikit-learn scorer for game points metric.

    Returns:
        Scorer object compatible with scikit-learn cross_val_score

    Usage:
        >>> from sklearn.model_selection import cross_val_score
        >>> scorer = make_game_points_scorer()
        >>> scores = cross_val_score(model, X, y, scoring=scorer, cv=5)
    """
    return make_scorer(
        calculate_game_points_array,
        greater_is_better=True,
    )


def score_prediction_breakdown(predicted: list[str], actual: list[str]) -> dict:
    """
    Score a prediction and provide detailed breakdown.

    Args:
        predicted: List of 3 predicted driver IDs
        actual: List of 3 actual driver IDs

    Returns:
        Dictionary with scoring breakdown:
        - total_points: Total points earned
        - exact_matches: Number of exact position matches
        - driver_matches: Number of correct drivers (any position)
        - position_details: List of per-position results
    """
    if len(predicted) != 3 or len(actual) != 3:
        raise ValueError("Both predicted and actual must contain exactly 3 drivers")

    total_points = 0
    exact_matches = 0
    driver_matches = 0
    position_details = []

    for i, pred_driver in enumerate(predicted):
        position_name = ["P1", "P2", "P3"][i]
        actual_driver = actual[i]

        if pred_driver == actual_driver:
            # Exact match
            total_points += 2
            exact_matches += 1
            driver_matches += 1
            position_details.append(
                {
                    "position": position_name,
                    "predicted": pred_driver,
                    "actual": actual_driver,
                    "result": "exact_match",
                    "points": 2,
                }
            )
        elif pred_driver in actual:
            # Correct driver, wrong position
            total_points += 1
            driver_matches += 1
            actual_position = ["P1", "P2", "P3"][actual.index(pred_driver)]
            position_details.append(
                {
                    "position": position_name,
                    "predicted": pred_driver,
                    "actual": actual_driver,
                    "result": "driver_match",
                    "points": 1,
                    "note": f"{pred_driver} was actually {actual_position}",
                }
            )
        else:
            # No match
            position_details.append(
                {
                    "position": position_name,
                    "predicted": pred_driver,
                    "actual": actual_driver,
                    "result": "no_match",
                    "points": 0,
                }
            )

    breakdown = {
        "total_points": total_points,
        "exact_matches": exact_matches,
        "driver_matches": driver_matches,
        "position_details": position_details,
        "accuracy_percent": (total_points / 6.0) * 100,
    }

    logger.info(
        f"Scoring breakdown: {total_points}/6 points "
        f"({exact_matches} exact, {driver_matches} drivers correct)"
    )

    return breakdown


if __name__ == "__main__":
    # Setup logging for testing
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Test examples
    print("Test 1: Perfect prediction")
    result = score_prediction_breakdown(["VER", "LEC", "HAM"], ["VER", "LEC", "HAM"])
    print(f"Score: {result['total_points']}/6\n")

    print("Test 2: Partial match")
    result = score_prediction_breakdown(["VER", "HAM", "LEC"], ["VER", "LEC", "HAM"])
    print(f"Score: {result['total_points']}/6\n")

    print("Test 3: No match")
    result = score_prediction_breakdown(["VER", "NOR", "PIA"], ["HAM", "LEC", "SAI"])
    print(f"Score: {result['total_points']}/6\n")
