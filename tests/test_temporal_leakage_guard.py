"""Unit tests for the temporal leakage guard and schema audit registry."""

import unittest
import pandas as pd

from early_warning.features.temporal_audit import (
    ColumnLifecycleStatus,
    TemporalLeakageError,
    get_audit_dataframe,
    is_column_allowed_at_t0,
    verify_zero_leakage,
)


class TestTemporalLeakageGuard(unittest.TestCase):
    """Test suite ensuring zero temporal or target leakage in feature spaces."""

    def setUp(self):
        """Build standard valid feature dataframe at T0."""
        self.valid_features_df = pd.DataFrame(
            {
                "freight_ratio": [0.15, 0.22, 0.08],
                "total_order_value": [120.50, 45.00, 310.00],
                "payment_installments": [1, 3, 10],
                "payment_type_boleto": [0, 1, 0],
                "approval_delay_hours": [0.45, 12.30, 0.10],
                "product_weight_g": [450, 1200, 3500],
                "is_interstate": [1, 0, 1],
                "days_to_next_holiday": [4, 18, 1],
                "is_holiday_eve": [0, 0, 1],
                "is_black_friday_week": [0, 0, 0],
            }
        )

    def test_valid_feature_matrix_passes(self):
        """Ensure a cleanly engineered T0 feature matrix passes without exception."""
        try:
            verify_zero_leakage(self.valid_features_df)
        except TemporalLeakageError as err:
            self.fail(f"verify_zero_leakage raised an unexpected error on valid data: {err}")

    def test_rejection_of_actual_delivery_date(self):
        """Assert that injecting order_delivered_customer_date raises TemporalLeakageError."""
        leaky_df = self.valid_features_df.copy()
        leaky_df["order_delivered_customer_date"] = [
            "2017-05-10 14:00:00",
            "2017-06-12 18:30:00",
            "2017-08-01 09:15:00",
        ]
        with self.assertRaises(TemporalLeakageError) as ctx:
            verify_zero_leakage(leaky_df)
        self.assertIn("TEMPORAL LEAKAGE DETECTED", str(ctx.exception))
        self.assertIn("order_delivered_customer_date", str(ctx.exception))

    def test_rejection_of_carrier_delivery_date(self):
        """Assert that injecting carrier hand-off timestamp raises TemporalLeakageError."""
        leaky_df = self.valid_features_df.copy()
        leaky_df["order_delivered_carrier_date"] = [
            "2017-05-02 10:00:00",
            "2017-06-03 11:30:00",
            "2017-07-25 16:15:00",
        ]
        with self.assertRaises(TemporalLeakageError) as ctx:
            verify_zero_leakage(leaky_df)
        self.assertIn("order_delivered_carrier_date", str(ctx.exception))

    def test_rejection_of_derived_delivery_delay_metrics(self):
        """Assert that synthetic post-T0 delay metrics raise TemporalLeakageError."""
        delay_feature_names = [
            "actual_delivery_days",
            "delivery_delay_days",
            "is_delayed",
            "is_late",
            "carrier_delay",
        ]
        for delay_col in delay_feature_names:
            leaky_df = self.valid_features_df.copy()
            leaky_df[delay_col] = [1, 0, 0]
            with self.assertRaises(
                TemporalLeakageError,
                msg=f"Failed to reject leakage feature '{delay_col}'",
            ):
                verify_zero_leakage(leaky_df)

    def test_rejection_of_target_variable(self):
        """Assert that review_score or binary target inside features raises TemporalLeakageError."""
        for target_col in ["review_score", "target", "is_negative_review"]:
            leaky_df = self.valid_features_df.copy()
            leaky_df[target_col] = [1, 0, 0]
            with self.assertRaises(
                TemporalLeakageError,
                msg=f"Failed to reject target variable '{target_col}' in feature matrix",
            ):
                verify_zero_leakage(leaky_df)

    def test_whitelist_enforcement(self):
        """Assert that unapproved unexpected columns are caught when a whitelist is provided."""
        whitelist = ["freight_ratio", "total_order_value", "payment_installments"]
        subset_df = self.valid_features_df[whitelist].copy()
        verify_zero_leakage(subset_df, allowed_whitelist=whitelist)

        # Inject an unexpected column not in whitelist
        subset_df["unexpected_column"] = [1, 2, 3]
        with self.assertRaises(TemporalLeakageError) as ctx:
            verify_zero_leakage(subset_df, allowed_whitelist=whitelist)
        self.assertIn("unexpected_column", str(ctx.exception))

    def test_audit_table_completeness(self):
        """Verify that the schema audit registry covers all 8 core tables with valid statuses."""
        audit_df = get_audit_dataframe()
        self.assertGreater(len(audit_df), 30)

        required_tables = {
            "olist.orders",
            "olist.order_items",
            "olist.order_payments",
            "olist.order_reviews",
            "olist.products",
            "olist.sellers",
            "olist.customers",
        }
        present_tables = set(audit_df["table_name"].unique())
        self.assertTrue(
            required_tables.issubset(present_tables),
            f"Audit table missing required tables: {required_tables - present_tables}",
        )

        valid_statuses = {status.value for status in ColumnLifecycleStatus}
        self.assertTrue(set(audit_df["status"].unique()).issubset(valid_statuses))

    def test_is_column_allowed_at_t0_helper(self):
        """Test the individual boolean check helper."""
        self.assertTrue(is_column_allowed_at_t0("freight_value"))
        self.assertTrue(is_column_allowed_at_t0("payment_installments"))
        self.assertTrue(is_column_allowed_at_t0("order_purchase_timestamp"))
        self.assertFalse(is_column_allowed_at_t0("order_delivered_customer_date"))
        self.assertFalse(is_column_allowed_at_t0("actual_delivery_days"))
        self.assertFalse(is_column_allowed_at_t0("review_score"))


if __name__ == "__main__":
    unittest.main()
