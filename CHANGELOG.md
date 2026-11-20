# Changelog

All notable changes to the Stage 4 Capstone Early Warning System are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.5.0] - 2026-11-20
### Added
- Comprehensive bilingual documentation suite: `report.md` (English) and `report.fa.md` (Persian) with AI usage disclosure.
- 10 formal Architecture Decision Records in `decisions.md` (ADR-001 through ADR-010).
- 1-page non-technical executive summary for operations leadership (`executive_summary.fa.md`).
- Flake8 linter configuration and Pytest warning filters (`.flake8`, `pytest.ini`).
- Final packaging metadata and distribution contracts.

### Changed
- Expanded `README.md` with complete CLI command reference, directory tree, and benchmark table.

---

## [0.4.0] - 2026-11-19
### Added
- Dual solution notebooks: `01_exploration.ipynb` (exploration, audit, API benchmarks, and leakage test) and `02_system.ipynb` (production pipeline, MLflow tracking, capacity evaluation, and priority export).
- Master single-command pipeline runner (`scripts/run_all.py` and `run_pipeline.ps1`) executing full cycle in < 45 seconds.
- Export of operational priority follow-up queue (`reports/priority_followup_orders.csv`).

---

## [0.3.5] - 2026-11-18
### Added
- Production inference engine (`early_warning.inference.predictor.EarlyWarningPredictor`) with automated schema validation and probability calibration.
- Unified command line interface (`early_warning.cli`) providing `audit`, `train`, `evaluate`, and `predict` commands.
- Synthetic end-to-end integration test (`tests/test_end_to_end_pipeline.py`) and inference unit tests (`tests/test_inference.py`).

---

## [0.3.0] - 2026-11-17
### Added
- MLflow tracking architecture (`early_warning.models.trainer`) logging runs, child runs for 2018 monthly cohorts, and diagnostic plots to local `mlruns.db`.
- Dual-tier model hierarchy: regularized Logistic Regression baseline and tuned LightGBM classifier.
- Serialized production models (`models/baseline_logistic.joblib`, `models/early_warning_lgbm.joblib`) and diagnostic curves in `reports/figures/`.

---

## [0.2.5] - 2026-11-16
### Added
- Operational capacity evaluator (`early_warning.models.capacity_evaluator`) supporting 5%, 10%, and 20% follow-up quotas.
- Empirical data leakage benchmark (`early_warning.models.leakage_benchmark`) measuring the deceptive gap (+24.31% capture rate surge from post-T0 attributes).
- Multi-period monthly walk-forward stability testing across 2018 test cohorts.
- Unit tests for capacity evaluation and monotonicity (`tests/test_models.py`).

---

## [0.2.0] - 2026-11-15
### Added
- PostgreSQL database dynamic view `olist.view_order_early_warning_features` and indexed materialized view `olist.mv_order_early_warning_features`.
- Execution plan benchmarks documenting sub-millisecond latencies (0.273 ms point lookup, 2.730 ms monthly filter, 8.750 ms full scan).
- Python database feature loader with holiday integration and unreviewed order quarantine (`early_warning.features.sql_features`).
- Database deployment automation script (`scripts/deploy_views.py`).

---

## [0.1.5] - 2026-11-14
### Added
- Comprehensive 45-column schema lifecycle audit registry (`early_warning.features.temporal_audit`).
- Automated runtime Zero-Temporal-Leakage guard (`verify_zero_leakage`) with regex interception of post-T0 features.
- Domain exception hierarchy (`TemporalLeakageError`) and forbidden column matching rules.
- Offline unit tests for temporal guard enforcement (`tests/test_temporal_leakage_guard.py`).

---

## [0.1.0] - 2026-11-13
### Added
- External API client for Brazilian National Statutory Holidays (`early_warning.data.external_api.BrasilAPIHolidayClient`) with persistent local disk caching.
- Benchmark client for Open-Meteo Historical Weather Archive (`OpenMeteoWeatherBenchmarkClient`) for empirical rejection comparison.
- Vectorized calculation of pre-T0 holiday proximity features: `days_to_next_holiday`, `is_holiday_eve`, and `is_black_friday_week`.
- Unit tests verifying deterministic holiday caching and boundary delta calculations (`tests/test_external_api.py`).

---

## [0.0.1] - 2026-11-12
### Added
- Modular package scaffolding: `src/early_warning/` (`config`, `data`, `features`, `models`, `inference`).
- Strict environment contracts: `requirements.txt` and `pyproject.toml` with console script entry point `early-warning`.
- Immutable configuration dataclass (`SystemConfig`) establishing default paths, database URL, and 10% capacity quota.
- Git repository initialization and `.gitignore` preventing data leakage.
