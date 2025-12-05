"""
Ranking evaluation metrics for F1 predictions.

Implements standard ranking metrics for evaluating top-3 predictions:
- NDCG@3: Normalized Discounted Cumulative Gain
- MAP@3: Mean Average Precision at 3
- Spearman correlation: Rank correlation coefficient
- Top-3 inclusion rate: Did we identify the right drivers?

Diagnostic metrics for understanding model weaknesses:
- Position-specific accuracy: P1/P2/P3 hit rates
- Near-miss analysis: How close are wrong predictions?
- Confidence calibration: Does confidence correlate with accuracy?

All metrics complement the game points scoring system (2/1/0) with
ranking-specific evaluation.
"""

import numpy as np
from scipy import stats

from src.evaluation.scoring import calculate_game_points


def ndcg_at_k(predicted: list[str], actual: list[str], k: int = 3) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at k.

    For top-3 prediction: measures how well we rank the top-3 drivers.
    Perfect prediction = 1.0, completely wrong = 0.0.

    Args:
        predicted: List of predicted driver codes in order
        actual: List of actual driver codes in order
        k: Number of positions to consider (default: 3)

    Returns:
        NDCG@k score between 0 and 1
    """
    if len(actual) < k or len(predicted) < k:
        return 0.0

    # Relevance scores: actual P1=k, P2=k-1, ..., Pk=1, others=0
    relevance = {}
    for i, driver in enumerate(actual[:k]):
        relevance[driver] = k - i  # P1=3, P2=2, P3=1

    # DCG: sum of relevance / log2(position + 1)
    dcg = 0.0
    for i, driver in enumerate(predicted[:k]):
        rel = relevance.get(driver, 0)
        dcg += rel / np.log2(i + 2)  # +2 because positions are 1-indexed

    # Ideal DCG (perfect ranking)
    ideal_relevances = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances))

    return dcg / idcg if idcg > 0 else 0.0


def map_at_k(predicted: list[str], actual: list[str], k: int = 3) -> float:
    """
    Calculate Mean Average Precision at k.

    Measures precision of top-k predictions considering order.
    Higher score = better precision at finding correct drivers in order.

    Args:
        predicted: List of predicted driver codes in order
        actual: List of actual driver codes in order
        k: Number of positions (default: 3)

    Returns:
        MAP@k score between 0 and 1
    """
    if len(actual) < k or len(predicted) < k:
        return 0.0

    actual_set = set(actual[:k])

    precisions = []
    hits = 0

    for i, driver in enumerate(predicted[:k]):
        if driver in actual_set:
            hits += 1
            precision_at_i = hits / (i + 1)
            precisions.append(precision_at_i)

    if not precisions:
        return 0.0

    return sum(precisions) / min(k, len(actual_set))


def spearman_correlation(
    predicted: list[str],
    actual: list[str],
) -> float:
    """
    Calculate Spearman rank correlation coefficient.

    Measures how well the predicted order matches actual order.
    1.0 = perfect correlation, 0 = no correlation, -1.0 = inverse.

    Args:
        predicted: Full predicted ranking (all drivers)
        actual: Full actual ranking (all drivers)

    Returns:
        Spearman rho between -1 and 1, or NaN if insufficient data
    """
    # Create rank mappings
    pred_ranks = {driver: i for i, driver in enumerate(predicted)}
    actual_ranks = {driver: i for i, driver in enumerate(actual)}

    # Find common drivers
    common = set(predicted) & set(actual)
    if len(common) < 3:
        return np.nan

    pred_r = [pred_ranks[d] for d in common]
    actual_r = [actual_ranks[d] for d in common]

    rho, _ = stats.spearmanr(pred_r, actual_r)
    return rho


def top_k_inclusion_rate(
    predicted: list[str],
    actual: list[str],
    k: int = 3,
) -> float:
    """
    Calculate what fraction of actual top-k drivers were predicted.

    Args:
        predicted: Predicted top-k drivers
        actual: Actual top-k drivers
        k: Number of positions

    Returns:
        Inclusion rate between 0 and 1
    """
    if len(actual) < k or len(predicted) < k:
        return 0.0

    actual_top_k = set(actual[:k])
    predicted_top_k = set(predicted[:k])

    intersection = actual_top_k & predicted_top_k
    return len(intersection) / k


def exact_match_rate(
    predicted: list[str],
    actual: list[str],
    k: int = 3,
) -> float:
    """
    Calculate what fraction of positions were exactly matched.

    Args:
        predicted: Predicted top-k drivers
        actual: Actual top-k drivers
        k: Number of positions

    Returns:
        Exact match rate between 0 and 1
    """
    if len(actual) < k or len(predicted) < k:
        return 0.0

    exact_matches = sum(1 for i in range(k) if predicted[i] == actual[i])
    return exact_matches / k


def evaluate_predictions(
    predicted: list[str],
    actual: list[str],
) -> dict:
    """
    Comprehensive evaluation of a top-3 prediction.

    Returns dict with all metrics:
    - game_points: 0-6 family game scoring
    - ndcg_3: NDCG@3
    - map_3: MAP@3
    - top3_inclusion: Top-3 inclusion rate
    - exact_match_rate: Exact position matches
    - spearman: Spearman correlation (if full rankings provided)

    Args:
        predicted: Predicted drivers (at least top-3)
        actual: Actual drivers (at least top-3)

    Returns:
        Dictionary with all evaluation metrics
    """
    return {
        "game_points": calculate_game_points(predicted[:3], actual[:3]),
        "ndcg_3": ndcg_at_k(predicted, actual, k=3),
        "map_3": map_at_k(predicted, actual, k=3),
        "top3_inclusion": top_k_inclusion_rate(predicted, actual, k=3),
        "exact_match_rate": exact_match_rate(predicted, actual, k=3),
        "spearman": spearman_correlation(predicted, actual),
    }


def evaluate_multiple_predictions(
    predictions: list[list[str]],
    actuals: list[list[str]],
) -> dict:
    """
    Evaluate multiple predictions and aggregate metrics.

    Args:
        predictions: List of predicted driver lists
        actuals: List of actual driver lists

    Returns:
        Dictionary with mean and std of all metrics
    """
    metrics = {
        "game_points": [],
        "ndcg_3": [],
        "map_3": [],
        "top3_inclusion": [],
        "exact_match_rate": [],
        "spearman": [],
    }

    for pred, actual in zip(predictions, actuals, strict=False):
        result = evaluate_predictions(pred, actual)
        for key in metrics:
            value = result[key]
            if not np.isnan(value):
                metrics[key].append(value)

    # Aggregate
    aggregated = {}
    for key, values in metrics.items():
        if values:
            aggregated[f"{key}_mean"] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))
        else:
            aggregated[f"{key}_mean"] = 0.0
            aggregated[f"{key}_std"] = 0.0

    aggregated["n_predictions"] = len(predictions)

    return aggregated


def format_evaluation_report(metrics: dict) -> str:
    """
    Format evaluation metrics as a human-readable report.

    Args:
        metrics: Dictionary from evaluate_predictions or evaluate_multiple_predictions

    Returns:
        Formatted string report
    """
    lines = ["Evaluation Results:", "-" * 40]

    if "game_points" in metrics:
        # Single prediction
        lines.append(f"  Game Points:     {metrics['game_points']}/6")
        lines.append(f"  NDCG@3:          {metrics['ndcg_3']:.3f}")
        lines.append(f"  MAP@3:           {metrics['map_3']:.3f}")
        lines.append(f"  Top-3 Inclusion: {metrics['top3_inclusion']:.1%}")
        lines.append(f"  Exact Match:     {metrics['exact_match_rate']:.1%}")
        if not np.isnan(metrics.get("spearman", np.nan)):
            lines.append(f"  Spearman:        {metrics['spearman']:.3f}")
    else:
        # Aggregated predictions
        lines.append(f"  Predictions:     {metrics.get('n_predictions', 0)}")
        lines.append(
            f"  Game Points:     {metrics['game_points_mean']:.2f} +/- {metrics['game_points_std']:.2f}"
        )
        lines.append(
            f"  NDCG@3:          {metrics['ndcg_3_mean']:.3f} +/- {metrics['ndcg_3_std']:.3f}"
        )
        lines.append(
            f"  MAP@3:           {metrics['map_3_mean']:.3f} +/- {metrics['map_3_std']:.3f}"
        )
        lines.append(f"  Top-3 Inclusion: {metrics['top3_inclusion_mean']:.1%}")
        lines.append(f"  Exact Match:     {metrics['exact_match_rate_mean']:.1%}")

    return "\n".join(lines)


# =============================================================================
# Diagnostic Metrics
# =============================================================================


def position_accuracy(
    predicted: list[str],
    actual: list[str],
    k: int = 3,
) -> dict[str, bool]:
    """
    Check if each position was predicted exactly correct.

    Args:
        predicted: Predicted top-k drivers in order
        actual: Actual top-k drivers in order
        k: Number of positions to check

    Returns:
        Dict with P1_correct, P2_correct, P3_correct booleans
    """
    result = {}
    for i in range(min(k, len(predicted), len(actual))):
        result[f"P{i + 1}_correct"] = predicted[i] == actual[i]
    return result


def position_accuracy_aggregate(
    predictions: list[list[str]],
    actuals: list[list[str]],
    k: int = 3,
) -> dict[str, float]:
    """
    Calculate position-specific accuracy rates across multiple predictions.

    Args:
        predictions: List of predicted driver lists
        actuals: List of actual driver lists
        k: Number of positions

    Returns:
        Dict with P1_accuracy, P2_accuracy, P3_accuracy as percentages
    """
    position_hits = {f"P{i + 1}": [] for i in range(k)}

    for pred, actual in zip(predictions, actuals, strict=False):
        pos_result = position_accuracy(pred, actual, k)
        for i in range(k):
            key = f"P{i + 1}_correct"
            if key in pos_result:
                position_hits[f"P{i + 1}"].append(pos_result[key])

    return {
        f"P{i + 1}_accuracy": np.mean(position_hits[f"P{i + 1}"])
        if position_hits[f"P{i + 1}"]
        else 0.0
        for i in range(k)
    }


def near_miss_analysis(
    predicted: list[str],
    actual: list[str],
    full_predicted_ranking: list[str] | None = None,
) -> dict:
    """
    Analyze near-misses: predictions that were close but not in top-3.

    Args:
        predicted: Predicted top-3 drivers
        actual: Actual top-3 drivers
        full_predicted_ranking: Full predicted ranking (optional, for P4 analysis)

    Returns:
        Dict with:
        - missed_drivers: Actual top-3 drivers we didn't predict
        - wrong_predictions: Our predictions that weren't in actual top-3
        - p4_was_top3: Whether our P4 prediction was actually in top-3
        - actual_positions_of_wrong: Where our wrong predictions actually finished
    """
    pred_set = set(predicted[:3])
    actual_set = set(actual[:3])

    missed = actual_set - pred_set  # Drivers in actual top-3 we missed
    wrong = pred_set - actual_set  # Drivers we predicted who weren't top-3

    result = {
        "missed_drivers": list(missed),
        "wrong_predictions": list(wrong),
        "n_missed": len(missed),
        "n_wrong": len(wrong),
    }

    # Check if P4 prediction was actually in top-3
    if full_predicted_ranking and len(full_predicted_ranking) > 3:
        p4_driver = full_predicted_ranking[3]
        result["p4_prediction"] = p4_driver
        result["p4_was_top3"] = p4_driver in actual_set

    return result


def near_miss_aggregate(
    predictions: list[list[str]],
    actuals: list[list[str]],
    full_rankings: list[list[str]] | None = None,
) -> dict:
    """
    Aggregate near-miss analysis across multiple predictions.

    Args:
        predictions: List of predicted top-3 lists
        actuals: List of actual top-3 lists
        full_rankings: List of full predicted rankings (optional)

    Returns:
        Dict with aggregated near-miss statistics
    """
    total_missed = 0
    total_wrong = 0
    p4_in_top3_count = 0
    p4_total = 0

    for i, (pred, actual) in enumerate(zip(predictions, actuals, strict=False)):
        full_rank = full_rankings[i] if full_rankings else None
        analysis = near_miss_analysis(pred, actual, full_rank)

        total_missed += analysis["n_missed"]
        total_wrong += analysis["n_wrong"]

        if "p4_was_top3" in analysis:
            p4_total += 1
            if analysis["p4_was_top3"]:
                p4_in_top3_count += 1

    n = len(predictions)
    return {
        "avg_missed_per_session": total_missed / n if n > 0 else 0,
        "avg_wrong_per_session": total_wrong / n if n > 0 else 0,
        "p4_was_actually_top3_rate": p4_in_top3_count / p4_total if p4_total > 0 else 0,
        "n_sessions": n,
    }


def confidence_calibration(
    predictions: list[dict],
    actuals: list[list[str]],
    confidence_bins: list[tuple[float, float]] | None = None,
) -> dict:
    """
    Analyze whether model confidence correlates with accuracy.

    Args:
        predictions: List of prediction dicts with 'top3' and 'predictions' keys
                    where predictions[i] has 'confidence' scores
        actuals: List of actual top-3 driver lists
        confidence_bins: List of (min, max) tuples for binning (default: low/medium/high)

    Returns:
        Dict with game points breakdown by confidence level
    """
    if confidence_bins is None:
        confidence_bins = [
            (0, 50, "low"),
            (50, 80, "medium"),
            (80, 100, "high"),
        ]

    bin_results = {label: [] for _, _, label in confidence_bins}

    for pred_dict, actual in zip(predictions, actuals, strict=False):
        if "predictions" not in pred_dict:
            continue

        # Get average confidence for this prediction
        confidences = [p.get("confidence", 50) for p in pred_dict["predictions"][:3]]
        avg_confidence = np.mean(confidences) if confidences else 50

        # Calculate game points for this prediction
        pred_top3 = pred_dict.get("top3", [])
        if len(pred_top3) >= 3 and len(actual) >= 3:
            points = calculate_game_points(pred_top3[:3], actual[:3])

            # Assign to bin
            for min_conf, max_conf, label in confidence_bins:
                if min_conf <= avg_confidence < max_conf:
                    bin_results[label].append(points)
                    break

    result = {}
    for _, _, label in confidence_bins:
        result[f"{label}_confidence_avg_points"] = (
            float(np.mean(bin_results[label])) if bin_results[label] else 0.0
        )
        result[f"{label}_confidence_count"] = len(bin_results[label])
    return result


class DiagnosticCollector:
    """
    Collects prediction results during CV for comprehensive diagnostics.

    Usage:
        collector = DiagnosticCollector()
        for session in sessions:
            pred_result = model.predict_top3(X, drivers)
            collector.add_prediction(pred_result, actual_top3)
        report = collector.get_report()
    """

    def __init__(self):
        self.predictions: list[list[str]] = []
        self.actuals: list[list[str]] = []
        self.full_rankings: list[list[str]] = []
        self.prediction_dicts: list[dict] = []
        self.game_points: list[int] = []

    def add_prediction(
        self,
        pred_result: dict,
        actual_top3: list[str],
    ) -> None:
        """
        Add a prediction result for later analysis.

        Args:
            pred_result: Dict from model.predict_top3() with 'top3', 'predictions', 'full_ranking'
            actual_top3: Actual top-3 drivers
        """
        pred_top3 = pred_result.get("top3", [])[:3]
        if len(pred_top3) < 3 or len(actual_top3) < 3:
            return

        self.predictions.append(pred_top3)
        self.actuals.append(actual_top3[:3])
        self.full_rankings.append(pred_result.get("full_ranking", pred_top3))
        self.prediction_dicts.append(pred_result)
        self.game_points.append(calculate_game_points(pred_top3, actual_top3[:3]))

    def get_report(self) -> dict:
        """
        Generate comprehensive diagnostic report.

        Returns:
            Dict with all metrics and diagnostics
        """
        if not self.predictions:
            return {"error": "No predictions collected"}

        # Core metrics
        report = {
            "n_sessions": len(self.predictions),
            "game_points_mean": float(np.mean(self.game_points)),
            "game_points_std": float(np.std(self.game_points)),
            "game_points_min": int(np.min(self.game_points)),
            "game_points_max": int(np.max(self.game_points)),
        }

        # Game points distribution
        report["game_points_distribution"] = {str(i): self.game_points.count(i) for i in range(7)}

        # Position-specific accuracy
        pos_acc = position_accuracy_aggregate(self.predictions, self.actuals)
        report.update(pos_acc)

        # Near-miss analysis
        near_miss = near_miss_aggregate(self.predictions, self.actuals, self.full_rankings)
        report.update(near_miss)

        # Standard ranking metrics
        ranking = evaluate_multiple_predictions(self.predictions, self.actuals)
        report["ndcg_3_mean"] = ranking["ndcg_3_mean"]
        report["map_3_mean"] = ranking["map_3_mean"]
        report["top3_inclusion_mean"] = ranking["top3_inclusion_mean"]
        report["exact_match_rate_mean"] = ranking["exact_match_rate_mean"]

        # Confidence calibration (if available)
        if self.prediction_dicts and "predictions" in self.prediction_dicts[0]:
            conf_cal = confidence_calibration(self.prediction_dicts, self.actuals)
            report.update(conf_cal)

        return report

    def format_report(self) -> str:
        """Format the diagnostic report as a human-readable string."""
        report = self.get_report()

        if "error" in report:
            return f"Error: {report['error']}"

        lines = [
            "",
            "=" * 60,
            "DIAGNOSTIC REPORT",
            "=" * 60,
            "",
            f"Sessions evaluated: {report['n_sessions']}",
            "",
            "GAME POINTS:",
            f"  Mean:  {report['game_points_mean']:.2f} +/- {report['game_points_std']:.2f}",
            f"  Range: {report['game_points_min']} - {report['game_points_max']}",
            "",
            "  Distribution:",
        ]

        for pts in range(7):
            count = report["game_points_distribution"].get(str(pts), 0)
            pct = count / report["n_sessions"] * 100 if report["n_sessions"] > 0 else 0
            bar = "#" * int(pct / 2)
            lines.append(f"    {pts} pts: {count:3d} ({pct:5.1f}%) {bar}")

        lines.extend(
            [
                "",
                "POSITION ACCURACY:",
                f"  P1: {report.get('P1_accuracy', 0):.1%}",
                f"  P2: {report.get('P2_accuracy', 0):.1%}",
                f"  P3: {report.get('P3_accuracy', 0):.1%}",
                "",
                "NEAR-MISS ANALYSIS:",
                f"  Avg drivers missed per session: {report.get('avg_missed_per_session', 0):.2f}",
                f"  P4 prediction was actually top-3: {report.get('p4_was_actually_top3_rate', 0):.1%}",
                "",
                "RANKING METRICS:",
                f"  NDCG@3:          {report.get('ndcg_3_mean', 0):.3f}",
                f"  MAP@3:           {report.get('map_3_mean', 0):.3f}",
                f"  Top-3 Inclusion: {report.get('top3_inclusion_mean', 0):.1%}",
                f"  Exact Match:     {report.get('exact_match_rate_mean', 0):.1%}",
            ]
        )

        # Confidence calibration if available
        if "high_confidence_avg_points" in report:
            lines.extend(
                [
                    "",
                    "CONFIDENCE CALIBRATION:",
                    f"  High (80-100%):   {report.get('high_confidence_avg_points', 0):.2f} pts "
                    f"({report.get('high_confidence_count', 0)} sessions)",
                    f"  Medium (50-80%):  {report.get('medium_confidence_avg_points', 0):.2f} pts "
                    f"({report.get('medium_confidence_count', 0)} sessions)",
                    f"  Low (0-50%):      {report.get('low_confidence_avg_points', 0):.2f} pts "
                    f"({report.get('low_confidence_count', 0)} sessions)",
                ]
            )

        lines.extend(["", "=" * 60])

        return "\n".join(lines)
