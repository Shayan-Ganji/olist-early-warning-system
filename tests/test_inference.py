"""Unit and integration tests for production inference engine and CLI."""

import unittest
import numpy as np
import pandas as pd

from early_warning.features.temporal_audit import TemporalLeakageError
from early_warning.inference.predictor import EarlyWarningPredictor


class TestInferenceEngine(unittest.TestCase):
    """Test suite ensuring inference pipeline operates with zero temporal leakage."""

    def setUp(self):
        self.predictor = EarlyWarningPredictor()

        # Build synthetic valid pre-T0 orders
        n_samples = 50
        self.synthetic_orders = pd.DataFrame(
            {
                "order_id": [f"order_{i:04d}" for i in range(n_samples)],
                "order_approved_at": ["2018-08-15 10:00:00"] * n_samples,
                "approval_delay_hours": np.random.uniform(0.1, 48.0, n_samples),
                "promised_delivery_days": np.random.uniform(5.0, 30.0, n_samples),
                "total_order_value": np.random.uniform(20.0, 500.0, n_samples),
                "max_installments": np.random.choice([1, 2, 3, 6, 10], n_samples),
                "payment_types_count": [1] * n_samples,
                "has_boleto_payment": np.random.choice([0, 1], n_samples),
                "has_voucher_payment": [0] * n_samples,
                "total_items_count": np.random.choice([1, 2, 3], n_samples),
                "unique_products_count": [1] * n_samples,
                "unique_sellers_count": [1] * n_samples,
                "total_items_price": np.random.uniform(20.0, 450.0, n_samples),
                "total_freight_value": np.random.uniform(10.0, 50.0, n_samples),
                "freight_ratio": np.random.uniform(0.05, 0.40, n_samples),
                "total_weight_g": np.random.uniform(200, 5000, n_samples),
                "total_volume_liters": np.random.uniform(1.0, 30.0, n_samples),
                "avg_photos_qty": np.random.choice([1, 2, 4], n_samples),
                "avg_desc_length": np.random.uniform(100, 1000, n_samples),
                "is_interstate": np.random.choice([0, 1], n_samples),
                "days_to_next_holiday": np.random.randint(1, 45, n_samples),
                "is_holiday_eve": [0] * n_samples,
                "is_black_friday_week": [0] * n_samples,
            }
        )

    def test_predict_orders_success(self):
        """Verify successful prediction, probability calibration, and priority ranking."""
        result = self.predictor.predict_orders(self.synthetic_orders, capacity_ratio=0.10)

        self.assertEqual(result.total_orders_scored, 50)
        # 10% of 50 = 5 flagged orders
        self.assertEqual(result.flagged_orders_count, 5)
        self.assertEqual(len(result.flagged_orders_df), 5)

        # Probabilities bounded between 0 and 1
        probs = result.ranked_orders_df["negative_review_probability"].values
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

        # Priority ranks are sequential 1 to N
        ranks = result.ranked_orders_df["priority_rank"].values
        self.assertListEqual(list(ranks), list(range(1, 51)))

        # Verify descending sort order
        self.assertTrue(np.all(np.diff(probs) <= 0.0001))

    def test_predict_rejects_leakage_features(self):
        """Verify that injecting a post-T0 feature into inference payload raises TemporalLeakageError."""
        leaky_payload = self.synthetic_orders.copy()
        leaky_payload["order_delivered_customer_date"] = "2018-08-22 14:00:00"

        with self.assertRaises(TemporalLeakageError) as ctx:
            self.predictor.predict_orders(leaky_payload)
        self.assertIn("TEMPORAL LEAKAGE DETECTED", str(ctx.exception))

    def test_predict_rejects_missing_features(self):
        """Verify that omitting a mandatory feature triggers a descriptive schema mismatch error."""
        incomplete_payload = self.synthetic_orders.drop(columns=["freight_ratio"])

        with self.assertRaises(ValueError) as ctx:
            self.predictor.predict_orders(incomplete_payload)
        self.assertIn("Schema mismatch", str(ctx.exception))
        self.assertIn("freight_ratio", str(ctx.exception))

    def test_empty_input_validation(self):
        """Verify that passing an empty DataFrame raises ValueError."""
        with self.assertRaises(ValueError):
            self.predictor.predict_orders(pd.DataFrame())


if __name__ == "__main__":
    unittest.main()
