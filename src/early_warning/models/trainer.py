"""Comprehensive Model Training, Capacity Evaluation, and MLflow Tracking Engine.

Trains both baseline (regularized logistic regression) and production (LightGBM)
models, performs temporal walk-forward backtesting, logs all runs and visual artifacts
to MLflow, and exports serialized production model artifacts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from early_warning.config import SystemConfig, default_config
from early_warning.features.sql_features import FeatureDataLoader
from early_warning.models.capacity_evaluator import (
    compute_capacity_metrics,
    evaluate_capacity_portfolio,
    evaluate_model_standard_metrics,
    evaluate_monthly_walkforward_stability,
)

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Orchestrates end-to-end model training, validation, MLflow logging, and artifact generation."""

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or default_config
        self.loader = FeatureDataLoader(self.config)
        self.models_dir = self.config.project_root / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir = self.config.project_root / "reports" / "figures"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Set up local MLflow SQLite tracking
        mlflow_db_path = self.config.project_root / "mlruns.db"
        self.mlflow_uri = f"sqlite:///{mlflow_db_path}"
        mlflow.set_tracking_uri(self.mlflow_uri)
        self.experiment_name = "olist_early_warning_system"
        mlflow.set_experiment(self.experiment_name)

    def load_and_split_data(
        self,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
        """Load features and split temporally at cutoff date (2018-03-01)."""
        X, y, meta = self.loader.prepare_dataset(use_materialized=True, enforce_leakage_guard=True)

        approved_dt = pd.to_datetime(meta["order_approved_at"], utc=True)
        cutoff_dt = pd.Timestamp(self.config.train_test_split_date, tz="UTC")

        train_mask = approved_dt < cutoff_dt
        test_mask = approved_dt >= cutoff_dt

        X_train, y_train = X[train_mask].copy(), y[train_mask].copy()
        X_test, y_test = X[test_mask].copy(), y[test_mask].copy()
        meta_train, meta_test = meta[train_mask].copy(), meta[test_mask].copy()

        logger.info(
            f"Temporal Split ({self.config.train_test_split_date}): "
            f"Train = {len(X_train):,} orders ({y_train.mean():.4f} neg rate), "
            f"Test = {len(X_test):,} orders ({y_test.mean():.4f} neg rate)"
        )
        return X_train, y_train, X_test, y_test, meta_train, meta_test

    def train_baseline_logistic(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        meta_test: pd.DataFrame,
    ) -> Dict[str, float]:
        """Train and evaluate regularized L2 Logistic Regression baseline with MLflow logging."""
        logger.info("Training Baseline Logistic Regression (Standardized)...")
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=self.config.random_seed,
                ),
            ),
        ])
        pipeline.fit(X_train, y_train)
        pred_prob_test = pipeline.predict_proba(X_test)[:, 1]

        with mlflow.start_run(run_name="01_baseline_logistic_regression"):
            mlflow.set_tags({
                "model_family": "logistic_regression",
                "tier": "baseline",
                "governance": "zero_leakage_enforced",
            })
            mlflow.log_params({
                "penalty": "l2",
                "C": 1.0,
                "class_weight": "balanced",
                "features_count": X_train.shape[1],
                "train_cutoff": self.config.train_test_split_date,
            })

            # Capacity metrics
            cap_metrics = evaluate_capacity_portfolio(y_test, pred_prob_test)
            for cap_name, m in cap_metrics.items():
                mlflow.log_metrics({
                    f"{cap_name}_capture_rate": m.capture_rate,
                    f"{cap_name}_precision": m.precision,
                    f"{cap_name}_lift": m.lift,
                    f"{cap_name}_threshold": m.threshold_score,
                })

            std_metrics = evaluate_model_standard_metrics(y_test, pred_prob_test)
            mlflow.log_metrics(std_metrics)

            # August 2018 final month metrics
            aug_mask = (
                pd.to_datetime(meta_test["order_approved_at"], utc=True)
                >= pd.Timestamp("2018-08-01", tz="UTC")
            ) & (
                pd.to_datetime(meta_test["order_approved_at"], utc=True)
                < pd.Timestamp("2018-09-01", tz="UTC")
            )
            y_aug = y_test[aug_mask].values
            p_aug = pred_prob_test[aug_mask]
            aug_cap = compute_capacity_metrics(y_aug, p_aug, capacity_ratio=0.10)
            mlflow.log_metrics({
                "august_2018_capture_rate_10pct": aug_cap.capture_rate,
                "august_2018_precision_10pct": aug_cap.precision,
                "august_2018_lift": aug_cap.lift,
            })

            # Save model
            model_path = self.models_dir / "baseline_logistic.joblib"
            joblib.dump(pipeline, model_path)
            mlflow.log_artifact(str(model_path))

        return {
            "capture_rate_10pct": cap_metrics["capacity_10pct"].capture_rate,
            "precision_10pct": cap_metrics["capacity_10pct"].precision,
            "lift_10pct": cap_metrics["capacity_10pct"].lift,
            "roc_auc": std_metrics["roc_auc"],
            "pr_auc": std_metrics["pr_auc"],
            "august_capture_rate_10pct": aug_cap.capture_rate,
        }

    def train_production_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        meta_test: pd.DataFrame,
    ) -> Dict[str, float]:
        """Train tuned LightGBM classifier with walk-forward backtest and artifact generation."""
        logger.info("Training Production LightGBM Classifier...")
        lgbm_params = {
            "n_estimators": 150,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": -1,
            "class_weight": "balanced",
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "random_state": self.config.random_seed,
            "verbose": -1,
        }

        clf = LGBMClassifier(**lgbm_params)
        clf.fit(X_train, y_train)
        pred_prob_test = clf.predict_proba(X_test)[:, 1]

        with mlflow.start_run(run_name="02_production_lightgbm"):
            mlflow.set_tags({
                "model_family": "lightgbm",
                "tier": "production",
                "governance": "zero_leakage_enforced",
            })
            mlflow.log_params(lgbm_params)
            mlflow.log_param("features_count", X_train.shape[1])
            mlflow.log_param("train_cutoff", self.config.train_test_split_date)

            # Capacity metrics across entire test set
            cap_metrics = evaluate_capacity_portfolio(y_test, pred_prob_test)
            for cap_name, m in cap_metrics.items():
                mlflow.log_metrics({
                    f"{cap_name}_capture_rate": m.capture_rate,
                    f"{cap_name}_precision": m.precision,
                    f"{cap_name}_lift": m.lift,
                    f"{cap_name}_threshold": m.threshold_score,
                })

            std_metrics = evaluate_model_standard_metrics(y_test, pred_prob_test)
            mlflow.log_metrics(std_metrics)

            # August 2018 final month evaluation
            aug_mask = (
                pd.to_datetime(meta_test["order_approved_at"], utc=True)
                >= pd.Timestamp("2018-08-01", tz="UTC")
            ) & (
                pd.to_datetime(meta_test["order_approved_at"], utc=True)
                < pd.Timestamp("2018-09-01", tz="UTC")
            )
            y_aug = y_test[aug_mask].values
            p_aug = pred_prob_test[aug_mask]
            aug_cap = compute_capacity_metrics(y_aug, p_aug, capacity_ratio=0.10)
            mlflow.log_metrics({
                "august_2018_capture_rate_10pct": aug_cap.capture_rate,
                "august_2018_precision_10pct": aug_cap.precision,
                "august_2018_lift": aug_cap.lift,
            })

            # Generate and log visual artifacts
            fig_paths = self._generate_visual_artifacts(
                y_test.values, pred_prob_test, clf, X_train.columns.tolist(), meta_test
            )
            for fig_p in fig_paths:
                mlflow.log_artifact(str(fig_p))

            # Save model artifact
            model_path = self.models_dir / "early_warning_lgbm.joblib"
            joblib.dump(clf, model_path)
            mlflow.log_artifact(str(model_path))

            # Temporal Walk-Forward Backtest Child Runs (Monthly Cohorts)
            test_df_eval = meta_test.copy()
            test_df_eval["is_negative_review"] = y_test.values
            test_df_eval["pred_prob"] = pred_prob_test
            monthly_stability_df = evaluate_monthly_walkforward_stability(
                test_df_eval,
                date_col="order_approved_at",
                y_true_col="is_negative_review",
                y_prob_col="pred_prob",
                capacity_ratio=0.10,
            )

            # Log monthly backtest child runs
            for _, row in monthly_stability_df.iterrows():
                month_str = row["cohort_month"]
                with mlflow.start_run(
                    run_name=f"backtest_cohort_{month_str}", nested=True
                ):
                    mlflow.set_tags({
                        "run_type": "temporal_backtest_child",
                        "cohort_month": month_str,
                    })
                    mlflow.log_metrics({
                        "order_count": row["order_count"],
                        "base_rate": row["base_rate"],
                        "capture_rate_10pct": row["capture_rate_10pct"],
                        "precision_10pct": row["precision_10pct"],
                        "lift_10pct": row["lift"],
                        "roc_auc": row["roc_auc"],
                        "pr_auc": row["pr_auc"],
                    })

        return {
            "capture_rate_10pct": cap_metrics["capacity_10pct"].capture_rate,
            "precision_10pct": cap_metrics["capacity_10pct"].precision,
            "lift_10pct": cap_metrics["capacity_10pct"].lift,
            "roc_auc": std_metrics["roc_auc"],
            "pr_auc": std_metrics["pr_auc"],
            "august_capture_rate_10pct": aug_cap.capture_rate,
            "monthly_stability_df": monthly_stability_df,
        }

    def _generate_visual_artifacts(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model: LGBMClassifier,
        feature_names: List[str],
        meta_test: pd.DataFrame,
    ) -> List[Path]:
        """Generate publication-ready diagnostic and operational charts."""
        created_paths: List[Path] = []

        # 1. Capacity Gain and Lift Curve (1% to 30%)
        caps = np.linspace(0.01, 0.30, 30)
        capture_rates = [compute_capacity_metrics(y_true, y_prob, cap).capture_rate * 100 for cap in caps]
        precisions = [compute_capacity_metrics(y_true, y_prob, cap).precision * 100 for cap in caps]

        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(caps * 100, capture_rates, "b-o", linewidth=2, label="Capture Rate @ Capacity (%)")
        ax1.axvline(x=10, color="r", linestyle="--", label="10% Operations Capacity Quota")
        ax1.set_xlabel("Intervention Capacity Quota (%)")
        ax1.set_ylabel("Capture Rate (% of Total Negative Reviews)", color="b")
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(caps * 100, precisions, "g-s", linewidth=2, label="Precision @ Capacity (%)")
        ax2.set_ylabel("Precision (% True Bad in Flagged)", color="g")

        fig.suptitle("Capacity-Constrained Operational Performance Curve (Clean Model)", fontsize=12)
        fig.tight_layout()
        path1 = self.reports_dir / "capacity_curve.png"
        fig.savefig(path1, dpi=200)
        plt.close(fig)
        created_paths.append(path1)

        # 2. Confusion Matrix Heatmap at 10% Quota
        cap10 = compute_capacity_metrics(y_true, y_prob, capacity_ratio=0.10)
        fig, ax = plt.subplots(figsize=(6, 5))
        cm_data = np.array([
            [cap10.total_orders - cap10.flagged_orders - (cap10.total_negatives - cap10.captured_negatives),
             cap10.false_alarms],
            [cap10.total_negatives - cap10.captured_negatives,
             cap10.captured_negatives]
        ])
        ax.imshow(cm_data, cmap="Blues", interpolation="nearest")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Not Contacted", "Contacted (Top 10%)"])
        ax.set_yticklabels(["Satisfied (3-5)", "Negative (1-2)"])
        for i in range(2):
            for j in range(2):
                text_color = "black" if cm_data[i, j] < 20000 else "white"
                ax.text(
                    j, i, f"{cm_data[i, j]:,}",
                    ha="center", va="center", color=text_color,
                    fontsize=11, fontweight="bold"
                )
        title_str = (
            f"Operational Confusion Matrix at 10% Quota\n"
            f"(Capture Rate: {cap10.capture_rate*100:.1f}%, Precision: {cap10.precision*100:.1f}%)"
        )
        ax.set_title(title_str)
        fig.tight_layout()
        path2 = self.reports_dir / "confusion_matrix_10pct.png"
        fig.savefig(path2, dpi=200)
        plt.close(fig)
        created_paths.append(path2)

        # 3. Precision-Recall Curve
        prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(rec_arr, prec_arr, "darkorange", linewidth=2, label="Clean LightGBM PR Curve")
        rand_label = f"Random Guess Baseline ({np.mean(y_true)*100:.1f}%)"
        ax.axhline(y=np.mean(y_true), color="navy", linestyle="--", label=rand_label)
        opt_label = (
            f"Operating Point @ 10% Quota "
            f"(Rec: {cap10.capture_rate*100:.1f}%, Prec: {cap10.precision*100:.1f}%)"
        )
        ax.scatter([cap10.capture_rate], [cap10.precision], color="red", s=100, zorder=5, label=opt_label)
        ax.set_xlabel("Recall (Capture Rate)")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve with Operating Point")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path3 = self.reports_dir / "pr_curve.png"
        fig.savefig(path3, dpi=200)
        plt.close(fig)
        created_paths.append(path3)

        # 4. Feature Importance Plot
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[-15:]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(range(len(sorted_idx)), importances[sorted_idx], color="steelblue", align="center")
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([feature_names[i] for i in sorted_idx])
        ax.set_xlabel("LightGBM Feature Importance (Split Count)")
        ax.set_title("Top 15 Most Informative Pre-T0 Features")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path4 = self.reports_dir / "feature_importance.png"
        fig.savefig(path4, dpi=200)
        plt.close(fig)
        created_paths.append(path4)

        return created_paths
