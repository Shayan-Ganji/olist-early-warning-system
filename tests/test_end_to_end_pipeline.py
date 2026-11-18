"""End-to-end integration test on synthetic dummy orders.

Validates that the entire early warning lifecycle:
Synthetic Orders -> Schema Assertion -> Holiday Merge -> Model Training ->
Capacity Evaluation -> Top-K Priority Queue Export -> Output Verification
completes deterministically without runtime error.
"""

import unittest
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from early_warning.data.external_api import BrasilAPIHolidayClient
from early_warning.features.sql_features import AUTHORIZED_FEATURE_COLUMNS
from early_warning.features.temporal_audit import verify_zero_leakage
from early_warning.models.capacity_evaluator import compute_capacity_metrics


class TestEndToEndPipeline(unittest.TestCase):
    """End-to-end integration test exercising the complete architecture on dummy data."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Generate 200 synthetic orders
        np.random.seed(42)
        n = 200
        self.dummy_orders = pd.DataFrame(
            {
                "order_id": [f"ord_{i:05d}" for i in range(n)],
                "customer_id": [f"cust_{i:05d}" for i in range(n)],
                "order_purchase_timestamp": pd.date_range("2017-06-01", periods=n, freq="4h"),
                "order_approved_at": pd.date_range("2017-06-01 01:00:00", periods=n, freq="4h"),
                "approval_delay_hours": np.random.uniform(0.1, 24.0, n),
                "promised_delivery_days": np.random.uniform(5.0, 30.0, n),
                "total_order_value": np.random.uniform(30.0, 450.0, n),
                "max_installments": np.random.choice([1, 2, 4, 8], n),
                "payment_types_count": [1] * n,
                "has_boleto_payment": np.random.choice([0, 1], n),
                "has_voucher_payment": [0] * n,
                "total_items_count": np.random.choice([1, 2], n),
                "unique_products_count": [1] * n,
                "unique_sellers_count": [1] * n,
                "total_items_price": np.random.uniform(20.0, 400.0, n),
                "total_freight_value": np.random.uniform(10.0, 50.0, n),
                "freight_ratio": np.random.uniform(0.05, 0.50, n),
                "total_weight_g": np.random.uniform(300, 4000, n),
                "total_volume_liters": np.random.uniform(1.0, 25.0, n),
                "avg_photos_qty": np.random.choice([1, 2, 3], n),
                "avg_desc_length": np.random.uniform(150, 800, n),
                "is_interstate": np.random.choice([0, 1], n),
                "is_negative_review": np.random.binomial(1, 0.15, n),
            }
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_complete_pipeline_flow(self):
        """Execute end-to-end flow from feature enrichment through capacity evaluation and ranking."""
        df = self.dummy_orders.copy()

        # Step 1: External Holiday Enrichment
        holiday_client = BrasilAPIHolidayClient()
        hol_feats = holiday_client.compute_holiday_features(df["order_approved_at"])
        for col in hol_feats.columns:
            df[col] = hol_feats[col].values

        self.assertIn("days_to_next_holiday", df.columns)
        self.assertIn("is_holiday_eve", df.columns)
        self.assertIn("is_black_friday_week", df.columns)

        # Step 2: Zero-Temporal-Leakage Assertion
        X = df[AUTHORIZED_FEATURE_COLUMNS].copy()
        verify_zero_leakage(X, allowed_whitelist=AUTHORIZED_FEATURE_COLUMNS)

        y = df["is_negative_review"].values

        # Step 3: Train-Test Split (150 train, 50 test)
        X_train, y_train = X.iloc[:150], y[:150]
        X_test, y_test = X.iloc[150:], y[150:]

        # Step 4: Model Training
        model = LGBMClassifier(n_estimators=20, learning_rate=0.1, random_state=42, verbose=-1)
        model.fit(X_train, y_train)
        pred_probs = model.predict_proba(X_test)[:, 1]

        # Step 5: Capacity Evaluation at 10% Quota
        cap_10 = compute_capacity_metrics(y_test, pred_probs, capacity_ratio=0.10)
        self.assertEqual(cap_10.total_orders, 50)
        self.assertEqual(cap_10.flagged_orders, 5)
        self.assertGreaterEqual(cap_10.capture_rate, 0.0)
        self.assertLessEqual(cap_10.capture_rate, 1.0)

        # Step 6: Priority Queue Generation and Export
        test_df = df.iloc[150:].copy()
        test_df["negative_review_probability"] = pred_probs
        ranked_df = test_df.sort_values(by="negative_review_probability", ascending=False).reset_index(drop=True)
        flagged_df = ranked_df.iloc[:5].copy()

        out_path = self.temp_path / "test_flagged.csv"
        flagged_df.to_csv(out_path, index=False)
        self.assertTrue(out_path.exists())
        self.assertEqual(len(pd.read_csv(out_path)), 5)


if __name__ == "__main__":
    unittest.main()
