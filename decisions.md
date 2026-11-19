# Architecture Decision Records (ADR)
## Early Warning Decision Engine for Customer Negative Reviews
**دانشجو:** شایان گنجی  
**آزمایشگاه:** فناوری های مالی داتین- ترم سوم  
**پایگاه‌داده هدف:** PostgreSQL 16.4 | دیتابیس `olist_db`  

---

### ADR-001: Strict Decision Boundary Moment $T_0$ Definition at `order_approved_at`

- **Status:** Accepted
- **Context:**
  The platform needs an early warning system to intercept orders likely to receive negative customer reviews (scores 1 or 2). In the order lifecycle, there are multiple timestamps: `order_purchase_timestamp`, `order_approved_at`, `order_delivered_carrier_date`, and `order_delivered_customer_date`. If $T_0$ is placed too late (e.g., carrier pickup), intervention options are severely constrained. If placed at purchase time before payment approval, orders that are ultimately rejected or canceled clog the priority follow-up pipeline.
- **Decision:**
  Formally define decision moment $T_0$ as the exact timestamp of order payment approval:
  $$T_0 \equiv \text{Timestamp}(\text{order\_approved})$$
  All features consumed by the model must be strictly available at or before this timestamp (`order_approved_at`).
- **Consequences:**
  - *Positive:* Legitimate operational intervention point. Merchant has not yet shipped the parcel, allowing platform support to contact sellers for expedited handling, prioritize carrier pickup, or proactively manage customer expectations.
  - *Negative:* Precludes using transit and carrier performance signals for the current order, capping raw offline ROC-AUC compared to naive leaky models.
- **Alternatives Considered:**
  - *Purchase Timestamp ($T_{\text{purchase}}$):* Rejected because boleto payments take 24–48 hours to settle. Contacting customers before payment clears wastes operations budget on abandoned checkouts.
  - *Carrier Dispatch Timestamp ($T_{\text{carrier}}$):* Rejected because the package has already left seller possession; operations cannot influence merchant packaging or carrier dispatch priority.

---

### ADR-002: External Data Selection (BrasilAPI Statutory Holidays) & Rejection of Weather/Inflation APIs

- **Status:** Accepted
- **Context:**
  The specification mandates evaluating at least two external live API sources, selecting one, proving zero temporal leakage, and documenting rejection rationales for the others.
- **Decision:**
  Select **BrasilAPI Brazilian Statutory Holidays** (`https://brasilapi.com.br/api/feriados/v1`) as the primary external data source. Reject **Open-Meteo Historical Weather**, **Banco Central do Brasil (BCB) IPCA Inflation**, and **ViaCEP Geolocation**.
- **Consequences:**
  - *Positive:*
    1. *Zero Temporal Leakage:* Statutory public holidays in Brazil are enacted by federal decrees (Lei Federal nº 662/1949 and nº 10.607/2002) months or years in advance. At moment $T_0$, the holiday schedule is 100% deterministic.
    2. *Direct Behavioral Mechanism:* Postal hubs (Correios) accumulate severe parcel backlogs during statutory holidays and Black Friday week. Customer delivery expectations are heightened prior to holidays.
    3. *Zero Runtime Fragility:* Holidays are cached locally to `data/external/holidays_br.json` for 100% offline reproducibility.
  - *Negative:* Holiday signals operate at a temporal granularity (days to next holiday) and do not capture localized route disruption.
- **Alternatives Evaluated & Rejection Rationale:**
  - *Open-Meteo Historical Weather Archive:* Evaluated and benchmarked. Rejected for two architectural reasons:
    1. *Spatial Aggregation Error:* Olist orders span 27 Brazilian states. Querying weather at $T_0$ for customer and seller municipalities requires querying dozens of weather stations per order, introducing high network overhead.
    2. *Temporal Lookahead Leakage:* Rain at moment $T_0$ does not affect delivery that occurs 12 days later. Querying weather during transit requires knowing the future shipping dates, which constitutes severe post-$T_0$ leakage.
  - *Banco Central do Brasil (BCB) IPCA Inflation API:* Evaluated and rejected. Macroeconomic inflation operates on monthly indices published with a 15–40 day reporting lag. It provides zero order-level discriminative power.
  - *ViaCEP Address API:* Evaluated and rejected. Redundant with the internal `olist.geolocation` table.

---

### ADR-003: In-Database Feature Engineering via Indexed Materialized View in PostgreSQL

- **Status:** Accepted
- **Context:**
  The platform requires feature generation inside PostgreSQL (`olist_db`) via Views or Procedures with a strict service level agreement: query latency $< 500\text{ ms}$ with execution plans documented.
- **Decision:**
  Implement a dual-database architecture:
  1. Dynamic View `olist.view_order_early_warning_features` for live online single-order inference.
  2. Materialized View `olist.mv_order_early_warning_features` indexed with unique B-Tree on `order_id` and secondary B-Trees on `order_approved_at` and `customer_state` for bulk model training and monthly cohort backtesting.
- **Consequences:**
  - *Positive:*
    - Point lookup for online scoring executes in **0.273 ms** (over 1,800x faster than the 500 ms SLA).
    - Monthly cohort backtest query executes in **2.730 ms**.
    - Full dataset scan across all 99,441 orders executes in **8.750 ms**.
    - Pushes aggregations (`SUM(payment_value)`, `COUNT(order_item_id)`, freight ratios) down to the PostgreSQL execution engine.
  - *Negative:* Materialized views require scheduled refresh (`REFRESH MATERIALIZED VIEW`) when new historical data is ingested.

---

### ADR-004: Defense-in-Depth Zero-Leakage Architecture (Schema Whitelist & Regex Pattern Guards)

- **Status:** Accepted
- **Context:**
  Data leakage is the most prevalent failure mode in machine learning systems. Naive practitioners frequently include columns like `order_delivered_customer_date` or calculate transit delays, which yields deceptively high offline metrics that collapse completely in live production.
- **Decision:**
  Implement a multi-layered defense-in-depth anti-leakage architecture:
  1. *Schema Audit Registry (`temporal_audit.py`):* Explicit mapping of all 45 schema attributes across all 8 tables into lifecycle statuses (`ALLOWED_AT_T0`, `FORBIDDEN_POST_T0`, `TARGET_ONLY`, `METADATA_IDENTIFIER`).
  2. *Runtime Regex Guard (`verify_zero_leakage`):* Automatically scans incoming DataFrames using regex patterns (`deliver.*date`, `actual.*deliver`, `carrier.*date`, `delay.*day`, `is.*delay`, `review.*`). Raises `TemporalLeakageError` immediately upon detecting forbidden patterns.
  3. *Unit Test Enforcement (`test_temporal_leakage_guard.py`):* Automated CI test verifying that injecting post-$T_0$ features fails immediately.
- **Consequences:**
  - *Positive:* Impossible for pipeline authors or downstream inference callers to inadvertently contaminate feature spaces.
  - *Negative:* Requires explicit registration for any newly engineered features.

---

### ADR-005: Capacity-Constrained Evaluation Metric (Capture Rate @ 10%) over Global ROC-AUC

- **Status:** Accepted
- **Context:**
  In real-world customer operations, budget is strictly constrained. The Olist customer operations team has capacity to contact at most 10% of confirmed orders for proactive follow-up. Standard ML metrics like global ROC-AUC, accuracy, or Macro-F1 evaluate the entire probability distribution, treating false positives at the 90th percentile identically to false positives at the 20th percentile.
- **Decision:**
  Adopt **Capture Rate at 10% Capacity** ($\text{Recall}_{\text{top 10\%}}$) as the primary optimization and governance metric, supplemented by secondary evaluations at 5% and 20% alternative quotas:
  $$\text{Capture Rate @ } K\% = \frac{\sum_{i \in \text{Top } K\%} y_i}{\sum_{\text{all } i} y_i}, \quad \text{Precision @ } K\% = \frac{\sum_{i \in \text{Top } K\%} y_i}{\lfloor N \times K\% \rfloor}, \quad \text{Lift} = \frac{\text{Precision @ } K\%}{\text{Base Rate}}$$
- **Consequences:**
  - *Positive:* Perfect alignment between model mathematical objective and business operational constraints. Guarantees the operations team maximizes negative review interception within their fixed 10% follow-up budget.
  - *Negative:* Metric is non-differentiable; model training still requires continuous surrogate losses (binary log-loss) with probability ranking.

---

### ADR-006: Temporal Cutoff Split (`2018-03-01`) vs. Random K-Fold Cross-Validation

- **Status:** Accepted
- **Context:**
  E-commerce and financial data exhibit strong temporal dependencies, macroeconomic inflation, logistics strikes (e.g., Brazilian postal strike in March 2018), and platform maturation.
- **Decision:**
  Enforce a strict temporal cutoff date of **`2018-03-01`**:
  - *Train Cohort:* September 2016 through February 2018 (58,759 orders).
  - *Test Cohort:* March 2018 through August 2018 (39,759 orders).
  - *Final Month Evaluation:* August 2018 evaluated standalone as required by specification.
- **Consequences:**
  - *Positive:* Simulates realistic forward-looking production deployment. Uncovers temporal concept drift and postal strike disruptions.
  - *Negative:* Results in lower apparent metrics than random k-fold cross-validation due to distribution shift.
- **Alternatives Considered:**
  - *Random 80/20 Train-Test Split:* Strictly rejected. Randomly shuffling orders leaks future macro patterns into past predictions, violating physical causality.

---

### ADR-007: MLflow for Lineage, Reproducibility, and Time-Series Multi-Cohort Backtesting

- **Status:** Accepted
- **Context:**
  Reproducibility and auditability are foundational requirements. Furthermore, evaluating time-series stability across monthly market regimes mirrors financial backtesting (tracking Drawdown and Sharpe equivalents across periods).
- **Decision:**
  Integrate **MLflow** with a local SQLite database backend (`sqlite:///mlruns.db`):
  - Track run parameters: Cutoff date, feature count, model family, class weighting, hyperparameters.
  - Track metrics: Capture Rate @ 5%, 10%, 20%, Precision @ 10%, PR-AUC, ROC-AUC, Brier score, August 2018 metrics.
  - Nested Child Runs: Log each monthly cohort (2018-03 through 2018-08) as an individual child run to track stability across logistics shocks.
  - Artifact Logging: Confusion matrices at 10% capacity, ROC/PR curves, feature importance plots, serialized model binaries.
- **Consequences:**
  - *Positive:* 100% offline reproducibility, complete audit lineage, zero external server dependencies.
  - *Negative:* Adds MLflow dependency to environment contracts.

---

### ADR-008: Quarantine Policy for 768 Unreviewed Orders vs. Groundless Optimistic/Pessimistic Imputation

- **Status:** Accepted
- **Context:**
  Out of 99,441 orders in Olist, 768 orders (0.77%) have no corresponding review record in `olist.order_reviews`. Naive workflows either impute them as positive reviews (score 5, assuming "no news is good news") or negative reviews (score 1, assuming "unhappy customers refused to respond").
- **Decision:**
  Quarantine unreviewed orders from supervised model training and evaluation. For inference on unreviewed live orders, the system scores them normally using catalog and payment features.
- **Consequences:**
  - *Positive:* Avoids label noise and ungrounded behavioral assumptions. Preserves empirical integrity.
  - *Negative:* Drops 0.77% of rows from training set (58,759 clean training orders retained).

---

### ADR-009: Dual-Tier Model Hierarchy (Interpretable Regularized Logistic Baseline + Tuned LightGBM)

- **Status:** Accepted
- **Context:**
  A production ML system requires an interpretable baseline to validate that complex models extract non-linear patterns rather than overfitting.
- **Decision:**
  Implement a dual-tier model hierarchy:
  1. *Tier 1 Baseline:* Standardized L2-Regularized Logistic Regression with balanced class weights. Outputs feature odds ratios.
  2. *Tier 2 Production:* Tuned Gradient Boosted Trees (LightGBM) with balanced class weights, handling feature interactions (e.g., interstate delivery $\times$ heavy freight).
- **Consequences:**
  - *Positive:* Proves LightGBM delivers genuine operational lift (+0.80% capture rate, +1.08% precision at 10% capacity, +0.017 PR-AUC) over linear models.
  - *Negative:* Managing two model training pathways in the codebase.

---

### ADR-010: Offline Fallback and Deterministic Caching for Zero Runtime External Dependency

- **Status:** Accepted
- **Context:**
  Live external HTTP APIs can experience network timeouts, rate limiting, or DNS failure during CI/CD test runs or air-gapped production deployments.
- **Decision:**
  Implement deterministic local disk caching in `data/external/holidays_br.json`. If network requests fail, the client loads from local cache; if cache is missing, hardcoded statutory federal holidays serve as an immutable fallback.
- **Consequences:**
  - *Positive:* 100% offline unit tests, 0 ms network latency during testing, guaranteed zero build failures.
  - *Negative:* Requires local storage (~5 KB) for cached holiday payloads.

