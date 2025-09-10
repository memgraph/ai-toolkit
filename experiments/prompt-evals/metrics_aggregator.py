"""
Metrics aggregation utilities for DeepEval results.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def aggregate_metrics(results) -> Optional[Dict[str, Any]]:
    """
    Compute and log aggregated metrics for all test cases.

    Args:
        results: DeepEval evaluation results

    Returns:
        Dictionary containing aggregated metrics or None if no results
    """
    if not results or not hasattr(results, "test_results"):
        logger.info("No results to aggregate metrics from.")
        return None

    metric_names = []
    metric_scores = {}

    # Collect all metric names
    for test_result in results.test_results:
        for metric_data in getattr(test_result, "metrics_data", []):
            if metric_data.name not in metric_names:
                metric_names.append(metric_data.name)

    # Initialize lists for each metric
    for name in metric_names:
        metric_scores[name] = []

    # Gather all scores
    for test_result in results.test_results:
        for metric_data in getattr(test_result, "metrics_data", []):
            if metric_data.score is not None:
                metric_scores[metric_data.name].append(metric_data.score)

    logger.info("Aggregated Metrics (Averages):")
    final_scores = []
    aggregated_metrics = {}

    for name in metric_names:
        scores = metric_scores[name]
        if scores:
            avg = np.mean(scores)
            logger.info(f"  {name}: {avg:.3f}")
            final_scores.append(avg)
            aggregated_metrics[name] = {
                "average": avg,
                "scores": scores,
                "count": len(scores),
            }
        else:
            logger.info(f"  {name}: No scores available")
            final_scores.append(0.0)
            aggregated_metrics[name] = {"average": 0.0, "scores": [], "count": 0}

    # Compute a final score as the mean of all metric averages
    if final_scores:
        overall_score = np.mean(final_scores)
        logger.info(f"  Final Score for the whole test run: {overall_score:.3f}")
        aggregated_metrics["overall_score"] = overall_score
    else:
        logger.info("No scores available to compute a final score.")
        aggregated_metrics["overall_score"] = 0.0

    return aggregated_metrics
