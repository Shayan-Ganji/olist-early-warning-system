"""PostgreSQL Feature Loader and External Holiday Enrichment Pipeline.

Extracts engineered features from PostgreSQL database views (SLA < 500ms),
enriches them with BrasilAPI statutory holidays, and validates against
the Zero-Temporal-Leakage guard.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import pandas as pd
from sqlalchemy import create_engine

from early_warning.config import SystemConfig, default_config
from early_warning.data.external_api import BrasilAPIHolidayClient
from early_warning.features.temporal_audit import verify_zero_leakage

logger = logging.getLogger(__name__)


# Definitive Whitelist of Permissible Features at T0
AUTHORIZED_FEATURE_COLUMNS = [
    "approval_delay_hours",
    "promised_delivery_days",
    "total_order_value",
    "max_installments",
    "payment_types_count",
    "has_boleto_payment",
    "has_voucher_payment",
    "total_items_count",
    "unique_products_count",
    "unique_sellers_count",
    "total_items_price",
    "total_freight_value",
    "freight_ratio",
    "total_weight_g",
    "total_volume_liters",
    "avg_photos_qty",
    "avg_desc_length",
    "is_interstate",
    "days_to_next_holiday",
    "is_holiday_eve",
    "is_black_friday_week",
]

METADATA_COLUMNS = [
    "order_id",
    "customer_id",
    "order_purchase_timestamp",
    "order_approved_at",
    "customer_state",
    "dominant_seller_state",
]


class FeatureDataLoader:
    """Loads engineered features from PostgreSQL and merges external holiday signals."""

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or default_config
        self.engine = create_engine(self.config.db_connection_url)
        self.holiday_client = BrasilAPIHolidayClient(
            cache_dir=self.config.external_data_dir
        )

    def load_dataset_from_db(self, use_materialized: bool = True) -> pd.DataFrame:
        """Load features directly from PostgreSQL view.

        Args:
            use_materialized: If True, queries the indexed materialized view (mv_order_early_warning_features).
                              If False, queries the dynamic view (view_order_early_warning_features).

        Returns:
            DataFrame containing raw feature and metadata columns from the database.
        """
        table_name = (
            "olist.mv_order_early_warning_features"
            if use_materialized
            else "olist.view_order_early_warning_features"
        )
        query = f"SELECT * FROM {table_name};"
        logger.info(f"Loading feature dataset from PostgreSQL table '{table_name}'...")
        df = pd.read_sql_query(query, self.engine)
        logger.info(f"Loaded {len(df):,} records from database.")
        return df

    def enrich_with_holidays(
        self, df: pd.DataFrame, timestamp_col: str = "order_approved_at"
    ) -> pd.DataFrame:
        """Merge leak-free statutory calendar features based on order_approved_at."""
        logger.info("Enriching features with BrasilAPI statutory holiday calendar...")
        holiday_feats = self.holiday_client.compute_holiday_features(df[timestamp_col])
        out_df = df.copy()
        for col in holiday_feats.columns:
            out_df[col] = holiday_feats[col].values
        return out_df

    def prepare_dataset(
        self,
        use_materialized: bool = True,
        enforce_leakage_guard: bool = True,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Complete feature preparation pipeline.

        1. Fetches data from PostgreSQL.
        2. Filters out unreviewed or unapproved orders (quarantined records).
        3. Enriches with external holiday calendar.
        4. Validates zero-leakage constraints.
        5. Separates features X, target y, and metadata.

        Returns:
            Tuple of (X, y, metadata_df)
        """
        raw_df = self.load_dataset_from_db(use_materialized=use_materialized)

        # 1. Quarantine orders without review_score (768 orders in Olist)
        reviewed_mask = raw_df["is_negative_review"].notna()
        df = raw_df[reviewed_mask].copy()
        df["is_negative_review"] = df["is_negative_review"].astype(int)

        # 2. Enrich with external holidays
        df = self.enrich_with_holidays(df)

        # Fill missing values for numerical safety
        df["freight_ratio"] = df["freight_ratio"].fillna(0.0)
        df["avg_photos_qty"] = df["avg_photos_qty"].fillna(0.0)
        df["avg_desc_length"] = df["avg_desc_length"].fillna(0.0)

        # Separate components
        X = df[AUTHORIZED_FEATURE_COLUMNS].copy()
        y = df["is_negative_review"].copy()
        metadata = df[METADATA_COLUMNS].copy()

        # 3. Enforce Temporal Leakage Guard
        if enforce_leakage_guard:
            verify_zero_leakage(X, allowed_whitelist=AUTHORIZED_FEATURE_COLUMNS)

        logger.info(
            f"Prepared feature matrix X shape: {X.shape}, target y base rate: "
            f"{y.mean():.4f} ({y.sum():,} negative reviews out of {len(y):,})"
        )
        return X, y, metadata
