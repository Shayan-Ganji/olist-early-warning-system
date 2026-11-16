"""Unit tests for capacity evaluator, operational quotas, and model metrics."""

import unittest
import numpy as np
import pandas as pd

from early_warning.models.capacity_evaluator import (
    compute_capacity_metrics,
    evaluate_capacity_portfolio,
    evaluate_model_standard_metrics,
    evaluate_monthly_walkforward_stability,
)


class TestCapacityEvaluator(unittest.TestCase):
    """Test suite ensuring operational capacity mathematics and stability metrics are rigorous."""

    def setUp(self):
        # 100 orders, 20 positive (bad reviews), base rate = 0.20
        self.y_true = np.array([1] * 20 + [0] * 80)
        # Probabilities: perfectly separated for the first 10, then mixed
        self.y_prob = np.concatenate([
            np.linspace(0.9, 0.8, 10),   # 10 true positives flagged with high prob
            np.linspace(0.4, 0.1, 10),   # remaining 10 positives with lower prob
            np.linspace(0.7, 0.05, 80),  # negatives
        ])

    def test_capacity_metrics_basic_calculation(self):
        """Verify capacity metrics arithmetic at 10% quota."""
        # At 10% capacity, quota = 10 orders
        metrics = compute_capacity_metrics(self.y_true, self.y_prob, capacity_ratio=0.10)
        self.assertEqual(metrics.total_orders, 100)
        self.assertEqual(metrics.flagged_orders, 10)
        self.assertEqual(metrics.total_negatives, 20)
        self.assertGreater(metrics.captured_negatives, 0)
        self.assertAlmostEqual(metrics.capture_rate, metrics.captured_negatives / 20.0)
        self.assertAlmostEqual(metrics.precision, metrics.captured_negatives / 10.0)
        self.assertAlmostEqual(metrics.lift, metrics.precision / 0.20)

    def test_capacity_monotonicity(self):
        """Verify that capture rate increases monotonically with higher operational capacity."""
        cap_5 = compute_capacity_metrics(self.y_true, self.y_prob, capacity_ratio=0.05)
        cap_10 = compute_capacity_metrics(self.y_true, self.y_prob, capacity_ratio=0.10)
        cap_20 = compute_capacity_metrics(self.y_true, self.y_prob, capacity_ratio=0.20)

        self.assertLessEqual(cap_5.capture_rate, cap_10.capture_rate)
        self.assertLessEqual(cap_10.capture_rate, cap_20.capture_rate)
        self.assertLessEqual(cap_5.flagged_orders, cap_10.flagged_orders)
        self.assertLessEqual(cap_10.flagged_orders, cap_20.flagged_orders)

    def test_portfolio_evaluation(self):
        """Verify portfolio evaluation across 5%, 10%, 20%."""
        portfolio = evaluate_capacity_portfolio(self.y_true, self.y_prob)
        self.assertIn("capacity_5pct", portfolio)
        self.assertIn("capacity_10pct", portfolio)
        self.assertIn("capacity_20pct", portfolio)

    def test_standard_metrics(self):
        """Verify standard statistical classification metrics."""
        std = evaluate_model_standard_metrics(self.y_true, self.y_prob)
        self.assertIn("roc_auc", std)
        self.assertIn("pr_auc", std)
        self.assertIn("brier_score", std)
        self.assertGreaterEqual(std["roc_auc"], 0.5)
        self.assertLessEqual(std["brier_score"], 1.0)

    def test_monthly_walkforward_stability(self):
        """Verify multi-period monthly backtesting table computation."""
        df = pd.DataFrame({
            "order_approved_at": (
                ["2018-03-15"] * 60
                + ["2018-04-15"] * 60
                + ["2018-05-15"] * 60
            ),
            "is_negative_review": np.random.binomial(1, 0.15, 180),
            "pred_prob": np.random.uniform(0.01, 0.99, 180),
        })
        stability_df = evaluate_monthly_walkforward_stability(
            df,
            date_col="order_approved_at",
            y_true_col="is_negative_review",
            y_prob_col="pred_prob",
            capacity_ratio=0.10,
        )
        self.assertEqual(len(stability_df), 3)
        self.assertListEqual(list(stability_df["cohort_month"]), ["2018-03", "2018-04", "2018-05"])
        self.assertIn("capture_rate_10pct", stability_df.columns)
        self.assertIn("lift", stability_df.columns)

    def test_empty_array_error(self):
        """Assert ValueError on empty arrays."""
        with self.assertRaises(ValueError):
            compute_capacity_metrics([], [])


if __name__ == "__main__":
    unittest.main()
