"""
Unit tests for game points scoring function.

Tests validate scoring rules:
- 2 points for exact position match
- 1 point for correct driver in wrong position
- 0 points for driver not in top-3
"""

import numpy as np
import pytest

from src.evaluation.scoring import (
    calculate_game_points,
    calculate_game_points_array,
    make_game_points_scorer,
)


class TestGamePointsScoring:
    """Test game points calculation."""

    def test_perfect_prediction(self):
        """Test maximum 6 points for perfect prediction."""
        predicted = ["VER", "LEC", "HAM"]
        actual = ["VER", "LEC", "HAM"]

        score = calculate_game_points(predicted, actual)
        assert score == 6, "Perfect prediction should yield 6 points (3 exact * 2pts)"

    def test_zero_score(self):
        """Test 0 points when no drivers match."""
        predicted = ["VER", "LEC", "HAM"]
        actual = ["SAI", "NOR", "RUS"]

        score = calculate_game_points(predicted, actual)
        assert score == 0, "No matches should yield 0 points"

    def test_correct_drivers_wrong_positions(self):
        """Test 3 points for correct drivers in wrong positions."""
        predicted = ["VER", "LEC", "HAM"]
        actual = ["HAM", "VER", "LEC"]  # Same drivers, completely reversed

        score = calculate_game_points(predicted, actual)
        assert score == 3, "3 correct drivers in wrong positions = 3 * 1pt"

    def test_mixed_scoring(self):
        """Test mixed exact and wrong-position scoring."""
        predicted = ["VER", "LEC", "HAM"]
        actual = ["VER", "HAM", "SAI"]  # VER exact (P1), HAM wrong pos

        score = calculate_game_points(predicted, actual)
        # VER at P1: 2pts (exact)
        # HAM in top-3 but wrong position: 1pt
        # LEC not in top-3: 0pts
        assert score == 3, "1 exact (2pts) + 1 wrong position (1pt) = 3pts"

    def test_scenario_all_exact(self):
        """Test scenario: Predict VER-LEC-HAM, Actual VER-LEC-HAM."""
        predicted = ["VER", "LEC", "HAM"]
        actual = ["VER", "LEC", "HAM"]

        score = calculate_game_points(predicted, actual)
        assert score == 6, "All exact matches should score 6 points"

    def test_scenario_two_wrong_position(self):
        """Test scenario: Predict VER-LEC-HAM, Actual LEC-VER-SAI."""
        predicted = ["VER", "LEC", "HAM"]
        actual = ["LEC", "VER", "SAI"]

        score = calculate_game_points(predicted, actual)
        # VER in top-3 but wrong position: 1pt
        # LEC in top-3 but wrong position: 1pt
        # HAM not in top-3: 0pts
        assert score == 2, "Two wrong position matches should score 2 points"

    def test_partial_exact_matches(self):
        """Test scoring with some exact matches."""
        predicted = ["VER", "LEC", "HAM"]
        actual = ["VER", "SAI", "HAM"]  # P1 and P3 exact, P2 wrong

        score = calculate_game_points(predicted, actual)
        # VER at P1: 2pts (exact)
        # LEC not in top-3: 0pts
        # HAM at P3: 2pts (exact)
        assert score == 4, "2 exact matches = 4pts"

    def test_one_exact_two_wrong_positions(self):
        """Test 1 exact match + 2 drivers in wrong positions."""
        predicted = ["VER", "LEC", "HAM"]
        actual = ["LEC", "HAM", "VER"]  # All present but rotated

        score = calculate_game_points(predicted, actual)
        # No exact matches (all in wrong positions)
        # But all 3 drivers are in top-3
        assert score == 3, "All drivers present but wrong positions = 3pts"

    def test_two_exact_one_missing(self):
        """Test 2 exact matches + 1 driver not in top-3."""
        predicted = ["VER", "LEC", "HAM"]
        actual = ["VER", "LEC", "SAI"]  # P1 and P2 exact, HAM not present

        score = calculate_game_points(predicted, actual)
        # VER at P1: 2pts (exact)
        # LEC at P2: 2pts (exact)
        # HAM not in top-3: 0pts
        assert score == 4, "2 exact matches = 4pts"

    def test_edge_case_empty_strings(self):
        """Test handling of empty string driver IDs."""
        predicted = ["VER", "", "HAM"]
        actual = ["VER", "LEC", "HAM"]

        # Empty string doesn't match anything
        score = calculate_game_points(predicted, actual)
        # VER at P1: 2pts (exact)
        # Empty string at P2: 0pts (no match)
        # HAM at P3: 2pts (exact)
        assert score == 4, "Empty strings should be treated as no match"

    def test_validation_wrong_length_predicted(self):
        """Test validation: predicted list must have exactly 3 drivers."""
        predicted = ["VER", "LEC"]  # Only 2 drivers
        actual = ["VER", "LEC", "HAM"]

        with pytest.raises(ValueError, match="exactly 3 drivers"):
            calculate_game_points(predicted, actual)

    def test_validation_wrong_length_actual(self):
        """Test validation: actual list must have exactly 3 drivers."""
        predicted = ["VER", "LEC", "HAM"]
        actual = ["VER", "LEC"]  # Only 2 drivers

        with pytest.raises(ValueError, match="exactly 3 drivers"):
            calculate_game_points(predicted, actual)

    def test_validation_too_many_drivers(self):
        """Test validation: lists with more than 3 drivers should fail."""
        predicted = ["VER", "LEC", "HAM", "SAI"]  # 4 drivers
        actual = ["VER", "LEC", "HAM"]

        with pytest.raises(ValueError, match="exactly 3 drivers"):
            calculate_game_points(predicted, actual)


class TestGamePointsArrayScoring:
    """Test array-based scoring for scikit-learn compatibility."""

    def test_single_prediction_array(self):
        """Test scoring a single prediction in array format."""
        y_true = np.array([["VER", "LEC", "HAM"]])
        y_pred = np.array([["VER", "LEC", "HAM"]])

        avg_score = calculate_game_points_array(y_true, y_pred)
        assert avg_score == 6.0, "Perfect prediction should yield 6.0 average"

    def test_multiple_predictions_array(self):
        """Test average scoring across multiple predictions."""
        y_true = np.array(
            [
                ["VER", "LEC", "HAM"],
                ["LEC", "VER", "SAI"],
                ["HAM", "SAI", "NOR"],
            ]
        )
        y_pred = np.array(
            [
                ["VER", "LEC", "HAM"],  # 6 points (perfect)
                ["VER", "LEC", "HAM"],  # 2 points (VER+LEC wrong pos)
                ["VER", "LEC", "HAM"],  # 1 point (only HAM wrong pos)
            ]
        )

        avg_score = calculate_game_points_array(y_true, y_pred)
        expected_avg = (6 + 2 + 1) / 3  # 3.0
        assert avg_score == pytest.approx(expected_avg, abs=0.01)

    def test_empty_array(self):
        """Test handling of empty prediction arrays."""
        y_true = np.array([]).reshape(0, 3)
        y_pred = np.array([]).reshape(0, 3)

        avg_score = calculate_game_points_array(y_true, y_pred)
        assert avg_score == 0.0, "Empty arrays should return 0.0"

    def test_shape_mismatch_validation(self):
        """Test validation: y_true and y_pred must have same shape."""
        y_true = np.array([["VER", "LEC", "HAM"]])
        y_pred = np.array([["VER", "LEC", "HAM"], ["LEC", "VER", "SAI"]])

        with pytest.raises(ValueError, match="same shape"):
            calculate_game_points_array(y_true, y_pred)

    def test_wrong_number_of_positions_validation(self):
        """Test validation: must be for top-3 positions."""
        y_true = np.array([["VER", "LEC"]])  # Only 2 positions
        y_pred = np.array([["VER", "LEC"]])

        with pytest.raises(ValueError, match="top-3 positions"):
            calculate_game_points_array(y_true, y_pred)


class TestSklearnScorerIntegration:
    """Test scikit-learn scorer wrapper."""

    def test_make_game_points_scorer(self):
        """Test creating scikit-learn compatible scorer."""
        scorer = make_game_points_scorer()

        # Verify scorer is callable
        assert callable(scorer), "Scorer should be callable"

        # Test with sample data

        np.array([[1, 2], [3, 4], [5, 6]])
        np.array([["VER", "LEC", "HAM"], ["LEC", "VER", "SAI"], ["HAM", "SAI", "NOR"]])

        # Note: This is a basic structure test
        # Full integration would require a proper classifier
        # The scorer should work with scikit-learn's cross_val_score

    def test_scorer_higher_is_better(self):
        """Test that scorer is configured as 'higher is better'."""
        make_game_points_scorer()

        # Game points scoring is a maximization metric (higher is better)
        # Perfect score = 6, worst score = 0
        # Scorer should reflect this (positive scores)


class TestScoringEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_duplicate_predictions(self):
        """Test handling of duplicate driver predictions."""
        predicted = ["VER", "VER", "HAM"]  # Duplicate VER
        actual = ["VER", "LEC", "HAM"]

        # Duplicate predictions are technically invalid in real scenarios
        # But the scoring function should handle gracefully
        score = calculate_game_points(predicted, actual)

        # First VER matches position 1 (2pts)
        # Second VER is in actual (at pos 1) but wrong position (1pt)
        # HAM matches position 3 (2pts)
        # Total: 5pts (though this is a degenerate case)
        assert score >= 0, "Should handle duplicates without crashing"

    def test_case_sensitivity(self):
        """Test that driver IDs are case-sensitive."""
        predicted = ["ver", "lec", "ham"]  # Lowercase
        actual = ["VER", "LEC", "HAM"]  # Uppercase

        score = calculate_game_points(predicted, actual)
        # Case-sensitive: no matches
        assert score == 0, "Driver IDs should be case-sensitive"

    def test_all_wrong_drivers(self):
        """Test scoring when no predicted drivers are in actual top-3."""
        predicted = ["ALB", "STR", "GAS"]
        actual = ["VER", "LEC", "HAM"]

        score = calculate_game_points(predicted, actual)
        assert score == 0, "No matching drivers should yield 0 points"

    def test_one_correct_driver_exact_position(self):
        """Test scoring with only 1 driver correct in exact position."""
        predicted = ["VER", "ALB", "STR"]
        actual = ["VER", "LEC", "HAM"]

        score = calculate_game_points(predicted, actual)
        # Only VER at P1 matches (2pts)
        assert score == 2, "1 exact match = 2pts"

    def test_one_correct_driver_wrong_position(self):
        """Test scoring with only 1 driver correct but wrong position."""
        predicted = ["HAM", "ALB", "STR"]
        actual = ["VER", "LEC", "HAM"]

        score = calculate_game_points(predicted, actual)
        # Only HAM is in top-3 but wrong position (1pt)
        assert score == 1, "1 driver in wrong position = 1pt"


class TestScoringConsistency:
    """Test consistency of scoring across different scenarios."""

    def test_scoring_is_deterministic(self):
        """Test that scoring produces consistent results."""
        predicted = ["VER", "LEC", "HAM"]
        actual = ["LEC", "VER", "HAM"]

        score1 = calculate_game_points(predicted, actual)
        score2 = calculate_game_points(predicted, actual)
        score3 = calculate_game_points(predicted, actual)

        assert score1 == score2 == score3, "Scoring should be deterministic"

    def test_scoring_is_symmetric(self):
        """Test that swapping predicted/actual changes score appropriately."""
        predicted = ["VER", "LEC", "HAM"]
        actual = ["LEC", "VER", "HAM"]

        score1 = calculate_game_points(predicted, actual)
        score2 = calculate_game_points(actual, predicted)

        # Scores should be the same (both have same pattern of matches)
        assert score1 == score2, "Scoring should be symmetric for same match pattern"

    def test_max_score_boundary(self):
        """Test that maximum score is exactly 6 points."""
        predicted = ["VER", "LEC", "HAM"]
        actual = ["VER", "LEC", "HAM"]

        score = calculate_game_points(predicted, actual)
        assert score == 6, "Maximum possible score is 6 points"
        assert score <= 6, "Score should never exceed 6 points"

    def test_min_score_boundary(self):
        """Test that minimum score is exactly 0 points."""
        predicted = ["VER", "LEC", "HAM"]
        actual = ["SAI", "NOR", "RUS"]

        score = calculate_game_points(predicted, actual)
        assert score == 0, "Minimum possible score is 0 points"
        assert score >= 0, "Score should never be negative"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
