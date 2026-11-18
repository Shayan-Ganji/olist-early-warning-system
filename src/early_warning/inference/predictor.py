"""Production Inference Engine and Top-K Priority Ranker.

Takes newly approved orders at moment T0, applies schema verification and
zero-leakage assertion, enriches with statutory calendar holidays, computes
negative review probabilities, and outputs the top K% priority intervention list.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd

from early_warning.config import SystemConfig, default_config
from early_warning.data.external_api import BrasilAPIHolidayClient
from early_warning.features.sql_features import AUTHORIZED_FEATURE_COLUMNS
from early_warning.features.temporal_audit import verify_zero_leakage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictionResult:
    """Output bundle containing ranked predictions and capacity quota metadata."""

    total_orders_scored: int
    flagged_orders_count: int
    capacity_ratio: float
    probability_threshold: float
    ranked_orders_df: pd.DataFrame
    flagged_orders_df: pd.DataFrame


class EarlyWarningPredictor:
    """Production predictor for scoring and ranking orders at decision moment T0."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        config: Optional[SystemConfig] = None,
    ):
        self.config = config or default_config
        self.model_path = Path(
            model_path or (self.config.project_root / "models" / "early_warning_lgbm.joblib")
        )
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {self.model_path}. Train model first via ModelTrainer."
            )
        self.model = joblib.load(self.model_path)
        self.holiday_client = BrasilAPIHolidayClient(
            cache_dir=self.config.external_data_dir
        )

    def predict_orders(
        self,
        orders_df: pd.DataFrame,
        capacity_ratio: float = 0.10,
    ) -> PredictionResult:
        """Score incoming orders, verify zero leakage, and return top K% priority list.

        Args:
            orders_df: DataFrame containing pre-T0 features or raw order fields.
            capacity_ratio: Intervention quota (defaults to 0.10 for top 10%).

        Returns:
            PredictionResult with ranked orders and priority follow-up queue.
        """
        if orders_df.empty:
            raise ValueError("Input DataFrame is empty. Cannot perform inference.")

        df = orders_df.copy()

        # If holiday features are not pre-computed, enrich them if timestamp is present
        if "days_to_next_holiday" not in df.columns:
            if "order_approved_at" not in df.columns:
                raise ValueError(
                    "Missing 'order_approved_at' timestamp required for statutory holiday calculation."
                )
            holiday_feats = self.holiday_client.compute_holiday_features(df["order_approved_at"])
            for c in holiday_feats.columns:
                df[c] = holiday_feats[c].values

        # Ensure all required features are present
        missing_features = [c for c in AUTHORIZED_FEATURE_COLUMNS if c not in df.columns]
        if missing_features:
            raise ValueError(
                f"Schema mismatch: missing {len(missing_features)} required feature(s): {missing_features}"
            )

        # Enforce Zero-Temporal-Leakage constraint on full incoming payload
        verify_zero_leakage(df)

        X = df[AUTHORIZED_FEATURE_COLUMNS].copy()

        # Impute any missing numerical values
        X = X.fillna(0.0)

        # Generate predicted probabilities
        pred_probs = self.model.predict_proba(X)[:, 1]
        df["negative_review_probability"] = pred_probs

        # Rank descending
        df_ranked = df.sort_values(by="negative_review_probability", ascending=False).reset_index(drop=True)
        df_ranked["priority_rank"] = np.arange(1, len(df_ranked) + 1)

        total_orders = len(df_ranked)
        budget = max(1, int(np.ceil(total_orders * capacity_ratio)))
        df_flagged = df_ranked.iloc[:budget].copy()
        threshold_score = float(df_flagged.iloc[-1]["negative_review_probability"])

        logger.info(
            f"Scored {total_orders:,} orders. Flagged top {budget:,} orders "
            f"({capacity_ratio * 100:.1f}% capacity) with probability cutoff >= {threshold_score:.4f}"
        )

        return PredictionResult(
            total_orders_scored=total_orders,
            flagged_orders_count=budget,
            capacity_ratio=capacity_ratio,
            probability_threshold=threshold_score,
            ranked_orders_df=df_ranked,
            flagged_orders_df=df_flagged,
        )
