"""Operational Capacity-Based Evaluation Module for Customer Support Quotas.

In production, the customer operations team can contact at most K% (e.g., 10%)
of confirmed orders at moment T0. This module calculates operational capture rates,
precision, lift, and time-series stability across monthly evaluation cohorts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


@dataclass(frozen=True)
class CapacityMetrics:
    """Operational performance metrics at a specific intervention capacity quota."""

    capacity_ratio: float
    total_orders: int
    flagged_orders: int
    total_negatives: int
    captured_negatives: int
    false_alarms: int
    capture_rate: float  # Recall / Sensitivity at top K%
    precision: float     # Positive Predictive Value at top K%
    lift: float          # Precision / Base Rate
    threshold_score: float  # Predicted probability cutoff for top K%

    def to_dict(self) -> Dict[str, Union[int, float]]:
        return {
            "capacity_ratio": self.capacity_ratio,
            "total_orders": self.total_orders,
            "flagged_orders": self.flagged_orders,
            "total_negatives": self.total_negatives,
            "captured_negatives": self.captured_negatives,
            "false_alarms": self.false_alarms,
            "capture_rate": round(self.capture_rate, 4),
            "precision": round(self.precision, 4),
            "lift": round(self.lift, 2),
            "threshold_score": round(self.threshold_score, 4),
        }


def compute_capacity_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_prob: Union[np.ndarray, pd.Series],
    capacity_ratio: float = 0.10,
) -> CapacityMetrics:
    """Compute capacity-based operational metrics for a specific quota ratio.

    Args:
        y_true: Ground truth binary target (1 for negative review score 1 or 2, 0 otherwise).
        y_prob: Predicted probabilities of negative review P(y=1|X).
        capacity_ratio: Target contact capacity quota (e.g., 0.10 for top 10%).

    Returns:
        CapacityMetrics object containing capture rate, precision, lift, and confusion counts.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob).astype(float)

    total_orders = len(y_true_arr)
    if total_orders == 0:
        raise ValueError("Cannot evaluate capacity metrics on empty arrays.")

    total_negatives = int(np.sum(y_true_arr))
    base_rate = total_negatives / total_orders if total_orders > 0 else 0.0

    # Number of orders operations can follow up with
    budget_count = max(1, int(np.ceil(total_orders * capacity_ratio)))

    # Sort descending by predicted probability
    descending_indices = np.argsort(-y_prob_arr)
    flagged_indices = descending_indices[:budget_count]

    threshold_score = float(y_prob_arr[flagged_indices[-1]])
    captured_negatives = int(np.sum(y_true_arr[flagged_indices]))
    false_alarms = budget_count - captured_negatives

    capture_rate = (
        (captured_negatives / total_negatives) if total_negatives > 0 else 0.0
    )
    precision = captured_negatives / budget_count
    lift = (precision / base_rate) if base_rate > 0 else 1.0

    return CapacityMetrics(
        capacity_ratio=capacity_ratio,
        total_orders=total_orders,
        flagged_orders=budget_count,
        total_negatives=total_negatives,
        captured_negatives=captured_negatives,
        false_alarms=false_alarms,
        capture_rate=capture_rate,
        precision=precision,
        lift=lift,
        threshold_score=threshold_score,
    )


def evaluate_capacity_portfolio(
    y_true: Union[np.ndarray, pd.Series],
    y_prob: Union[np.ndarray, pd.Series],
    capacity_ratios: Iterable[float] = (0.05, 0.10, 0.20),
) -> Dict[str, CapacityMetrics]:
    """Evaluate performance across a portfolio of operational capacity quotas."""
    results = {}
    for cap in capacity_ratios:
        key = f"capacity_{int(cap * 100)}pct"
        results[key] = compute_capacity_metrics(y_true, y_prob, capacity_ratio=cap)
    return results


def evaluate_model_standard_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_prob: Union[np.ndarray, pd.Series],
) -> Dict[str, float]:
    """Compute standard statistical classification metrics alongside operational capacity."""
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob).astype(float)

    roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))
    pr_auc = float(average_precision_score(y_true_arr, y_prob_arr))
    brier = float(brier_score_loss(y_true_arr, y_prob_arr))
    base_rate = float(np.mean(y_true_arr))

    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier, 4),
        "base_rate": round(base_rate, 4),
    }


def evaluate_monthly_walkforward_stability(
    df: pd.DataFrame,
    date_col: str = "order_approved_at",
    y_true_col: str = "is_negative_review",
    y_prob_col: str = "pred_prob",
    capacity_ratio: float = 0.10,
) -> pd.DataFrame:
    """Evaluate temporal walk-forward stability across consecutive monthly cohorts.

    Directly analogous to multi-period financial backtesting, this tracks
    whether model capture rate and precision remain stable across changing
    logistics regimes (e.g. postal strikes, seasonal spikes).
    """
    df_eval = df.copy()
    df_eval["eval_month"] = pd.to_datetime(df_eval[date_col], utc=True).dt.strftime("%Y-%m")

    monthly_records = []
    for month, group in df_eval.groupby("eval_month"):
        y_t = group[y_true_col].values
        y_p = group[y_prob_col].values
        if len(y_t) < 50:
            continue

        cap_metrics = compute_capacity_metrics(y_t, y_p, capacity_ratio=capacity_ratio)
        std_metrics = evaluate_model_standard_metrics(y_t, y_p)

        monthly_records.append(
            {
                "cohort_month": month,
                "order_count": cap_metrics.total_orders,
                "negative_count": cap_metrics.total_negatives,
                "base_rate": std_metrics["base_rate"],
                "flagged_orders": cap_metrics.flagged_orders,
                "captured_negatives": cap_metrics.captured_negatives,
                f"capture_rate_{int(capacity_ratio*100)}pct": cap_metrics.capture_rate,
                f"precision_{int(capacity_ratio*100)}pct": cap_metrics.precision,
                "lift": cap_metrics.lift,
                "roc_auc": std_metrics["roc_auc"],
                "pr_auc": std_metrics["pr_auc"],
                "brier_score": std_metrics["brier_score"],
            }
        )

    return pd.DataFrame(monthly_records)
