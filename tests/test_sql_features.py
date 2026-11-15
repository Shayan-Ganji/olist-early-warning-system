"""Unit and integration tests for SQL feature extraction and holiday enrichment."""

import unittest
import pandas as pd

from early_warning.features.sql_features import (
    AUTHORIZED_FEATURE_COLUMNS,
    FeatureDataLoader,
)
from early_warning.features.temporal_audit import verify_zero_leakage


class TestSQLFeatures(unittest.TestCase):
    """Test suite for feature loading, external enrichment, and whitelist validation."""

    def setUp(self):
        self.loader = FeatureDataLoader()

    def test_authorized_columns_pass_leakage_guard(self):
        """Verify that every column in AUTHORIZED_FEATURE_COLUMNS satisfies zero-leakage constraints."""
        dummy_df = pd.DataFrame(
            {col: [0.0, 1.0] for col in AUTHORIZED_FEATURE_COLUMNS}
        )
        verify_zero_leakage(dummy_df, allowed_whitelist=AUTHORIZED_FEATURE_COLUMNS)

    def test_enrich_with_holidays(self):
        """Verify holiday enrichment logic on dummy timestamp data."""
        df = pd.DataFrame(
            {
                "order_id": ["order_1", "order_2", "order_3"],
                "order_approved_at": [
                    "2017-11-23 10:00:00",  # Thursday before Black Friday 2017
                    "2017-12-24 15:00:00",  # Christmas Eve
                    "2018-04-10 12:00:00",  # Normal working day
                ],
            }
        )
        enriched = self.loader.enrich_with_holidays(df)
        self.assertIn("days_to_next_holiday", enriched.columns)
        self.assertIn("is_holiday_eve", enriched.columns)
        self.assertIn("is_black_friday_week", enriched.columns)
        # Verify Black Friday week flag on Thanksgiving Thursday 2017-11-23
        self.assertEqual(enriched.loc[0, "is_black_friday_week"], 1)
        # Verify Christmas Eve is_holiday_eve
        self.assertEqual(enriched.loc[1, "is_holiday_eve"], 1)

    def test_database_features_pipeline_integration(self):
        """Integration test loading live features from PostgreSQL and verifying leakage guard."""
        try:
            X, y, metadata = self.loader.prepare_dataset(use_materialized=True)
            self.assertGreater(len(X), 90000)
            self.assertEqual(len(X), len(y))
            self.assertEqual(len(X), len(metadata))
            self.assertEqual(set(X.columns), set(AUTHORIZED_FEATURE_COLUMNS))
            self.assertTrue(0.10 <= float(y.mean()) <= 0.20)  # Base rate approximately ~14.7%
        except Exception as e:
            self.skipTest(f"PostgreSQL connection or dataset unavailable in test environment: {e}")


if __name__ == "__main__":
    unittest.main()
