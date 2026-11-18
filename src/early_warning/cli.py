"""Command Line Interface (CLI) for Early Warning Decision Engine.

Provides unified commands for:
- Database deployment and EXPLAIN plan benchmarking
- Model training, walk-forward backtesting, and MLflow logging
- Empirical Data Leakage benchmarking and deceptive gap measurement
- Batch scoring and top-K priority ranking for customer operations
- Schema lifecycle auditing and zero-leakage inspection
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from early_warning.features.temporal_audit import get_audit_dataframe
from early_warning.inference.predictor import EarlyWarningPredictor
from early_warning.models.leakage_benchmark import run_leakage_benchmark
from early_warning.models.trainer import ModelTrainer


def cmd_audit(args: argparse.Namespace) -> int:
    """Print the complete schema lifecycle audit table."""
    df_audit = get_audit_dataframe()
    print("\n" + "=" * 90)
    print("SCHEMA LIFECYCLE AUDIT: T0 AVAILABILITY REGISTRY")
    print("=" * 90)
    print(df_audit.to_string(index=False))
    print("=" * 90)
    print(f"Total Columns Audited: {len(df_audit)}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train baseline and production models, log to MLflow, and save artifacts."""
    print("\n" + "=" * 90)
    print("EARLY WARNING SYSTEM: MODEL TRAINING & MLFLOW TRACKING PIPELINE")
    print("=" * 90)
    trainer = ModelTrainer()
    X_train, y_train, X_test, y_test, meta_train, meta_test = trainer.load_and_split_data()

    print("\n[1/2] Training Baseline Logistic Regression...")
    res_lr = trainer.train_baseline_logistic(X_train, y_train, X_test, y_test, meta_test)
    cap_lr = res_lr['capture_rate_10pct'] * 100
    prec_lr = res_lr['precision_10pct'] * 100
    print(f"  -> Capture Rate @ 10%: {cap_lr:.2f}% | Precision: {prec_lr:.2f}% | ROC-AUC: {res_lr['roc_auc']:.4f}")

    print("\n[2/2] Training Production LightGBM Classifier...")
    res_lgb = trainer.train_production_lightgbm(X_train, y_train, X_test, y_test, meta_test)
    cap_lgb = res_lgb['capture_rate_10pct'] * 100
    prec_lgb = res_lgb['precision_10pct'] * 100
    print(f"  -> Capture Rate @ 10%: {cap_lgb:.2f}% | Precision: {prec_lgb:.2f}% | ROC-AUC: {res_lgb['roc_auc']:.4f}")
    aug_cap = res_lgb['august_capture_rate_10pct'] * 100
    print(f"  -> August 2018 (Final Month) Capture Rate @ 10%: {aug_cap:.2f}%")

    print("\n" + "=" * 90)
    print("MONTHLY WALK-FORWARD STABILITY (TEST COHORTS MARCH - AUGUST 2018)")
    print("=" * 90)
    stab_df = res_lgb["monthly_stability_df"]
    cols_to_print = ["cohort_month", "order_count", "base_rate", "capture_rate_10pct", "precision_10pct", "lift"]
    print(stab_df[cols_to_print].to_string(index=False))
    print("=" * 90)
    print("Training pipeline complete! Artifacts and models saved.")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Run full operational capacity evaluation and empirical leakage benchmark."""
    print("\n" + "=" * 90)
    print("EARLY WARNING SYSTEM: OPERATIONAL & EMPIRICAL LEAKAGE BENCHMARK")
    print("=" * 90)
    bench_result = run_leakage_benchmark()
    print("\nSIDE-BY-SIDE PERFORMANCE BENCHMARK (CLEAN VS. LEAKY):")
    print("-" * 90)
    print(bench_result.comparison_table.to_string(index=False))
    print("-" * 90)
    print("\nDECEPTIVE GAP ANALYSIS:")
    print(bench_result.deceptive_narrative)
    print("=" * 90)
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    """Score orders from CSV/Parquet and output the top K% priority intervention queue."""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}", file=sys.stderr)
        return 1

    print(f"Loading input orders from {input_path}...")
    if input_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    predictor = EarlyWarningPredictor()
    result = predictor.predict_orders(df, capacity_ratio=args.capacity)

    output_path = Path(args.output or "reports/priority_followup_orders.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.flagged_orders_df.to_csv(output_path, index=False)

    print("\n" + "=" * 90)
    print(f"PREDICTION COMPLETE: Scored {result.total_orders_scored:,} orders.")
    print(f"Flagged Top {result.flagged_orders_count:,} orders at {args.capacity * 100:.1f}% capacity.")
    print(f"Probability Cutoff Threshold: >= {result.probability_threshold:.4f}")
    print(f"Priority follow-up queue saved to: {output_path}")
    print("=" * 90)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="early-warning",
        description="Early Warning Decision Engine for Customer Negative Reviews (Olist Capstone)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: audit
    subparsers.add_parser("audit", help="Display temporal lifecycle schema audit registry.")

    # Subcommand: train
    subparsers.add_parser("train", help="Train models, perform backtest, and log to MLflow.")

    # Subcommand: evaluate
    subparsers.add_parser("evaluate", help="Run capacity evaluation and clean vs. leaky benchmark.")

    # Subcommand: predict
    pred_parser = subparsers.add_parser("predict", help="Score orders and output top K% follow-up list.")
    pred_parser.add_argument("--input", "-i", required=True, help="Path to input orders file (CSV or Parquet).")
    pred_parser.add_argument("--output", "-o", default=None, help="Path to save priority flagged orders CSV.")
    pred_parser.add_argument(
        "--capacity", "-c", type=float, default=0.10,
        help="Operational capacity quota (default 0.10)."
    )

    return parser


def main() -> None:
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "audit":
        sys.exit(cmd_audit(args))
    elif args.command == "train":
        sys.exit(cmd_train(args))
    elif args.command == "evaluate":
        sys.exit(cmd_evaluate(args))
    elif args.command == "predict":
        sys.exit(cmd_predict(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
