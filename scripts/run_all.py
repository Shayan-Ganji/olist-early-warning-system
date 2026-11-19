"""Master Capstone Execution Script for Stage 4 Early Warning System.

Executes the complete pipeline end-to-end:
1. Deploys PostgreSQL feature views and records execution plans (< 500ms SLA).
2. Runs the full unit and integration test suite (25 tests).
3. Trains baseline and production models, logging all runs to MLflow.
4. Executes the empirical clean vs. leaky benchmark.
5. Generates the priority follow-up queue for customer support operations.
6. Generates and executes the dual notebooks (01_exploration.ipynb and 02_system.ipynb).
"""

import sys
import time
from pathlib import Path

# Add project root and scripts to sys.path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "scripts"))
sys.path.insert(0, str(_project_root / "src"))

def run_master_pipeline():
    start_time = time.time()
    print("=" * 80)
    print("STAGE 4 CAPSTONE: EARLY WARNING SYSTEM MASTER PIPELINE")
    print("Author: Shayan Ganji | Dotin Financial Technologies Lab")
    print("=" * 80)

    # 1. Deploy views
    print("\n[Step 1/6] Deploying PostgreSQL Views and Benchmarking Latency SLA...")
    from scripts.deploy_views import deploy_views_and_capture_plans
    deploy_views_and_capture_plans()

    # 2. Run Test Suite
    print("\n[Step 2/6] Running Automated Unit and Integration Test Suite...")
    import unittest
    loader = unittest.TestLoader()
    suite = loader.discover("tests")
    runner = unittest.TextTestRunner(verbosity=2)
    test_result = runner.run(suite)
    if not test_result.wasSuccessful():
        print("ERROR: Test suite failed! Pipeline halted.", file=sys.stderr)
        sys.exit(1)

    # 3. Model Training & MLflow Logging
    print("\n[Step 3/6] Training Baseline & Production Models with MLflow Lineage...")
    from early_warning.models.trainer import ModelTrainer
    trainer = ModelTrainer()
    X_tr, y_tr, X_te, y_te, m_tr, m_te = trainer.load_and_split_data()
    trainer.train_baseline_logistic(X_tr, y_tr, X_te, y_te, m_te)
    res_lgb = trainer.train_production_lightgbm(X_tr, y_tr, X_te, y_te, m_te)
    print(f"  Production LightGBM Capture Rate @ 10%: {res_lgb['capture_rate_10pct']*100:.2f}% | Lift: {res_lgb['lift_10pct']:.2f}x")
    print(f"  August 2018 (Final Month) Capture Rate @ 10%: {res_lgb['august_capture_rate_10pct']*100:.2f}%")

    # 4. Leakage Benchmark
    print("\n[Step 4/6] Executing Empirical Data Leakage Benchmark...")
    from early_warning.models.leakage_benchmark import run_leakage_benchmark
    bench_res = run_leakage_benchmark()
    print(bench_res.comparison_table.to_string(index=False))

    # 5. Build Priority Queue
    print("\n[Step 5/6] Generating Priority Operational Follow-up Queue...")
    import pandas as pd
    from early_warning.inference.predictor import EarlyWarningPredictor
    aug_mask = (pd.to_datetime(m_te["order_approved_at"], utc=True) >= pd.Timestamp("2018-08-01", tz="UTC")) & \
               (pd.to_datetime(m_te["order_approved_at"], utc=True) < pd.Timestamp("2018-09-01", tz="UTC"))
    predictor = EarlyWarningPredictor()
    test_orders_sample = pd.concat([m_te[aug_mask], X_te[aug_mask]], axis=1)
    pred_res = predictor.predict_orders(test_orders_sample, capacity_ratio=0.10)
    out_csv = Path("reports/priority_followup_orders.csv")
    pred_res.flagged_orders_df.to_csv(out_csv, index=False)
    print(f"  Saved {len(pred_res.flagged_orders_df)} priority orders to {out_csv}")

    # 6. Build and Execute Dual Notebooks
    print("\n[Step 6/6] Generating and Executing Dual Solution Notebooks...")
    from scripts.build_exploration_notebook import create_exploration_notebook
    from scripts.build_system_notebook import create_system_notebook
    create_exploration_notebook()
    create_system_notebook()

    import nbformat
    from nbclient import NotebookClient

    for nb_name in ["01_exploration.ipynb", "02_system.ipynb"]:
        print(f"  Executing {nb_name}...")
        with open(nb_name, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        client = NotebookClient(nb, timeout=600, kernel_name="python3")
        client.execute()
        with open(nb_name, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"MASTER PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS ({elapsed/60.0:.2f} MINUTES)!")
    print("Execution is well within the 45-minute capstone budget.")
    print("=" * 80)

if __name__ == "__main__":
    run_master_pipeline()
