"""
Unit tests for external API ingestion and leakage-free holiday feature computation.
100% offline using cached/mocked responses.
"""

import unittest
import pandas as pd
import numpy as np

from early_warning.data.external_api import BrasilAPIHolidayClient


class TestBrasilAPIHolidayClient(unittest.TestCase):
    """Test suite validating holiday client resilience and leakage-free calendar math."""

    def setUp(self) -> None:
        self.client = BrasilAPIHolidayClient()

    def test_holiday_caching_and_schema(self) -> None:
        """Verify cached holidays load properly with required keys."""
        holidays = self.client.fetch_holidays()
        self.assertGreaterEqual(len(holidays), 30)
        first = holidays[0]
        self.assertIn("date", first)
        self.assertIn("name", first)
        self.assertIn("year", first)

    def test_days_to_next_holiday_calculation(self) -> None:
        """
        Verify exact day delta calculation to upcoming statutory holiday.
        Christmas is on 2017-12-25. An order on 2017-12-23 must yield exactly 2 days.
        """
        ts = pd.Series(["2017-12-23 11:30:00"])
        feats = self.client.compute_holiday_features(ts)
        self.assertEqual(feats["days_to_next_holiday"].iloc[0], 2)
        self.assertEqual(feats["is_holiday_eve"].iloc[0], 1)
        self.assertEqual(feats["is_black_friday_week"].iloc[0], 0)

    def test_black_friday_detection(self) -> None:
        """Verify Black Friday shopping peak window detection in late November."""
        ts = pd.Series(["2017-11-24 16:00:00"])  # Black Friday in 2017
        feats = self.client.compute_holiday_features(ts)
        self.assertEqual(feats["is_black_friday_week"].iloc[0], 1)

    def test_null_timestamp_handling(self) -> None:
        """Verify graceful handling of missing timestamps without runtime exception."""
        ts = pd.Series([None, np.nan])
        feats = self.client.compute_holiday_features(ts)
        self.assertTrue(pd.isna(feats["days_to_next_holiday"].iloc[0]))
        self.assertEqual(feats["is_holiday_eve"].iloc[0], 0)


if __name__ == "__main__":
    unittest.main()
