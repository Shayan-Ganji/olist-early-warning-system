# Early Warning System Architecture & Operational Decision Report
## Predictive Interception of Customer Negative Reviews under Fixed Operational Capacity
**دانشجو:** شایان گنجی  
**آزمایشگاه:** فناوری های مالی داتین- ترم سوم  
**پایگاه‌داده هدف:** PostgreSQL 16.4 | دیتابیس `olist_db`  
**محیط اجرا:** Python 3.11, PostgreSQL 16, MLflow 3.18, LightGBM 4.6  

---

**Documentation Links:** [English Version](report.md) | [Persian Version](report.fa.md) | [Executive Summary](executive_summary.fa.md) | [Architecture Decisions (decisions.md)](decisions.md) | [Exploration Notebook](01_exploration.ipynb) | [System Notebook](02_system.ipynb)

---

## Executive Abstract

In e-commerce marketplace platforms, customer dissatisfaction culminating in negative reviews (ratings of 1 or 2 stars) drives severe churn and long-term brand impairment. Proactive customer support operations have a fixed capacity constraint: labor budgets permit proactive contact with at most **10% of confirmed orders**. 

The fundamental engineering challenge of this Capstone is physical and temporal: **all predictions must be generated at the exact moment of payment authorization ($T_0$, equivalent to `order_approved_at`), prior to merchant dispatch and carrier hand-off.** In historical datasets such as Olist (`olist_db`), tables store completed orders where post-fulfillment outcomes (actual transit durations, postal delays, carrier hand-off dates) are backfilled. Introducing even one such attribute creates severe data leakage, resulting in artificial validation scores that collapse completely in live production.

This early warning decision engine addresses this constraint via 21 strictly pre-$T_0$ features, in-database PostgreSQL feature generation executing in **under 3 ms** (far exceeding the 500 ms SLA), external calendar enrichment via Brazilian statutory holidays (BrasilAPI), experiment lineage tracking via MLflow, and runtime regex leakage guards. In production backtests, our LightGBM model achieves a **21.20% Capture Rate at 10% Capacity** with **28.77% Precision** and an **Operational Lift of 2.12x** over random calling. On the final unseen evaluation month (August 2018), it sustains a **21.01% Capture Rate** with **22.36% Precision**. The complete system reproduces from scratch via a single command in under **45 seconds**.

---

## 1. Physical Order Lifecycle & Strict Decision Boundary ($T_0$)

### 1.1 Chronological States and Intervention Windows
The physical journey of an order on the Olist platform traverses distinct chronological phases:

$$\text{Purchase Click} \xrightarrow{\quad} \text{Payment Approval } (T_0) \xrightarrow{\quad} \text{Carrier Hand-off} \xrightarrow{\quad} \text{Customer Delivery} \xrightarrow{\quad} \text{Review Survey}$$

Engineering analysis demonstrated that the only actionable operational window is at payment approval ($T_0$):
* **Why not before payment approval?** In Brazil, a large volume of transactions settle via cash vouchers (Boleto Bancário), which take 24 to 48 hours to clear. Many orders are abandoned or rejected prior to approval. Contacting buyers before payment clearance squanders operational quota on non-existent transactions.
* **Why not after carrier hand-off?** Once `order_delivered_carrier_date` is stamped, the package has physically left the merchant's custody and entered the national postal network (Correios). At this juncture, customer support cannot expedite merchant dispatch, switch courier tiers, or inspect package integrity.

Hence, moment $T_0$ (`order_approved_at`) was established as the immutable decision boundary. All model features are strictly restricted to information available at or before this timestamp.

### 1.2 Target Formulation & Handling of Unreviewed Orders
Across all 99,441 orders in `olist.orders`, review status distribution in `olist.order_reviews` is as follows (`01_exploration.ipynb` Cell 2):
* Score 1 (Severe Grievance): 11,424 orders (11.51%)
* Score 2 (Dissatisfied): 3,151 orders (3.18%)
* **Target Positive Class ($y = 1$, Scores 1 or 2): 14,575 orders (14.69% base rate)**
* Score 3 (Neutral): 8,179 orders (8.24%)
* Score 4 (Satisfied): 19,142 orders (19.29%)
* Score 5 (Delighted): 57,328 orders (57.78%)
* **Unreviewed Orders:** 768 orders (0.77%)

**The 768 Unreviewed Orders Dilemma:**  
A common mistake is either naively dropping these orders or imputing them as 0 (satisfied). Assuming satisfaction introduces sample selection bias, while imputing 1 penalizes merchants unfairly. We implemented a **Quarantine Strategy** (ADR-008): these 768 orders are excluded from supervised training to prevent label noise, but the production inference pipeline is required to score *all* incoming orders at $T_0$, regardless of whether a review is eventually recorded.

---

## 2. External API Investigation: Failed Hypotheses & The Final Pivot

A key component of this Capstone was exploring, testing, and scientifically justifying an external live data signal. We tested multiple intuitive hypotheses that failed empirical or temporal validity before arriving at our final solution (`01_exploration.ipynb` Cell 6).

### 2.1 Attempt 1: Historical Weather Archive (Open-Meteo API) — Rejected
* **Initial Hypothesis:** Extreme meteorological events (heavy precipitation, windstorms) disrupt highway transport across Brazil, delaying courier trucks and triggering negative reviews.
* **Empirical Test:** We implemented `OpenMeteoWeatherBenchmarkClient` and queried daily precipitation and wind speed across Brazil's top fulfillment hubs (São Paulo, Rio de Janeiro, Curitiba) for 2017–2018.
* **Why it Failed Empirically and Temporally:**
  1. **Temporal Horizon Paradox:** At moment $T_0$, delivery is 10 to 25 days into the future. Weather forecast skill decays after 3 to 5 days. Precipitation on the day of payment approval has zero correlation with whether it rained when the truck crossed an interstate mountain pass 18 days later.
  2. **Severe Missingness in Remote Municipalities:** Olist spans 4,119 Brazilian municipalities. Most rural destinations lack local meteorological stations, resulting in > 35% missing values or noisy spatial interpolations.
  3. **Zero Model Lift:** Feature importance in LightGBM was < 0.5% split share, and adding weather features yielded 0.00% improvement in Capture Rate @ 10%.

### 2.2 Attempt 2: Macroeconomic Inflation & Consumer Confidence (IBGE) — Rejected
* **Initial Hypothesis:** Rapid shifts in monthly inflation (IPCA index) or consumer confidence diminish buyer patience for delayed packages.
* **Why it Failed:**
  1. **Lookahead Publication Lag Trap:** Official inflation metrics are published monthly with a 10 to 15-day lag *after* month-end. Using March's inflation rate for an order placed on March 5 is severe lookahead leakage; that figure was not yet compiled or published on March 5.
  2. **Granularity Mismatch:** Macroeconomic trends evolve across quarters and cannot explain intra-day order dissatisfaction within the same city.

### 2.3 The Validated Winner: Brazilian Statutory Holidays (BrasilAPI)
We implemented [BrasilAPIHolidayClient](file:///d:/Lab/term%203/stage-4-shayan-ganji/src/early_warning/data/external_api.py) to extract national statutory holidays.
* **Why this Source is 100% Scientifically Sound & Leak-Free:**
  1. **Legally Predetermined in Advance (Zero Leakage):** Federal statutory holidays in Brazil (Carnival, Good Friday, Tiradentes, Labor Day, Corpus Christi, Independence Day, Republic Day, Christmas) are established by federal legislation years in advance. Their calendar dates are known with 100% certainty at $T_0$.
  2. **Direct Causal Friction in Postal Logistics:** The Brazilian postal service (Correios) and regional line-haul carriers completely shut down during statutory holidays. Parcels approved 1 to 3 days prior to holiday clusters sit idle in sorting hubs, generating severe backlog waves.
  3. **Strong Empirical Signal:** The negative review rate for orders approved during holiday eves rose to 18.4% (vs. 14.1% during standard working weeks).
  4. **Offline Resilience:** All holiday calendars are cached locally in [data/external/holidays_br.json](file:///d:/Lab/term%203/stage-4-shayan-ganji/data/external/holidays_br.json), guaranteeing zero runtime network dependency.

---

## 3. In-Database Feature Engineering & Sub-Millisecond Query SLAs

Per system requirements, feature extraction was executed entirely within PostgreSQL `olist_db` under a strict 500 ms SLA (`02_system.ipynb` Cell 2).

### 3.1 The Naive Multi-Table Join Bottleneck
In early development, a naive dynamic SQL query joined 8 relational tables (`orders`, `order_items`, `customers`, `sellers`, `order_payments`, `products`, `geolocation`) with complex window aggregations. This query took **1,240 ms** to execute across the dataset, violating the 500 ms SLA.

### 3.2 Architectural Resolution: Materialized View with Composite B-Trees
We replaced the monolithic query with a dual-view architecture in [sql/views/01_create_feature_views.sql](file:///d:/Lab/term%203/stage-4-shayan-ganji/sql/views/01_create_feature_views.sql):
* **Indexed Materialized View (`olist.mv_order_early_warning_features`):** Pre-aggregates basket, payment, and product features, fortified with B-Tree indexes on `order_id`, `order_approved_at`, and `customer_state`.
* **Dynamic View (`olist.view_order_early_warning_features`):** Available for real-time online lookups without refresh dependencies.

### 3.3 Query Execution Plan Benchmarks (`EXPLAIN ANALYZE BUFFERS`)
Execution plans documented in [sql/views/feature_query_plan.txt](file:///d:/Lab/term%203/stage-4-shayan-ganji/sql/views/feature_query_plan.txt) confirm dramatic latency reductions:
* **Plan 1: Point Lookup (Online Order Scoring):** **0.273 ms** (1,831x faster than 500 ms SLA).
* **Plan 2: Monthly Cohort Filter (August 2018 - 6,523 orders):** **2.730 ms** (183x faster than SLA).
* **Plan 3: Full Table Scan (99,441 orders):** **8.750 ms** (57x faster than SLA).

---

## 4. Why MLflow in the Capstone (and not Stages 1–3)

A core architectural question: **Why was MLflow introduced specifically for the Capstone rather than earlier stages?**

### 4.1 Evolution Across Project Stages
* **Stage 1 (Data Factory):** Focused strictly on streaming HTTP ingestion, schema contracts, quarantine partitions, and deterministic hashing. No models were trained.
* **Stage 2 (Database Performance):** Focused on PostgreSQL internal mechanics: execution plans, B-Tree indexes, stored procedures, and analytical views.
* **Stage 3 (Credit Risk ML):** A static cross-sectional classification task evaluated on a single train-test split. Saving a joblib artifact and a lightweight dictionary of metrics was sufficient.
* **Stage 4 (Capstone):** A **longitudinal operational decision engine** operating across rolling temporal cohorts where model stability over time is paramount.

### 4.2 MLflow's Architectural Role in Stage 4
MLflow was configured to run 100% offline using a local SQLite backend ([mlruns.db](file:///d:/Lab/term%203/stage-4-shayan-ganji/mlruns.db)):
1. **Parent-Child Run Hierarchy for Time-Series Cohort Backtesting:**  
   The primary model trained on pre-March 2018 data is logged as the **Parent Run**. Then, each of the 6 subsequent evaluation months (March through August 2018) is logged as an independent **Child Run** recording its specific capture rate, precision, operational lift, and sample size. This enables operations to monitor time-series performance stability across time, analogous to quantitative financial backtesting (`02_system.ipynb` Cell 3).
2. **Capacity Portfolio Tracking:**  
   Rather than tracking a single global ROC-AUC scalar, MLflow logs performance curves and confusion matrices across 5%, 10%, and 20% operational contact quotas.
3. **Immutable Audit Trail for Data Leakage:**  
   Both the clean model and the leaky contaminated model are registered within the same experiment (`olist_early_warning_system`), permanently logging the +24.31% deceptive gap as an empirical benchmark.

---

## 5. Temporal Data Leakage Audit & The Deceptive Performance Gap

### 5.1 Empirical Clean vs. Leaky Benchmark
To demonstrate the danger of temporal contamination, module [early_warning.models.leakage_benchmark](file:///d:/Lab/term%203/stage-4-shayan-ganji/src/early_warning/models/leakage_benchmark.py) trained two identical LightGBM models on pre-March 2018 orders and evaluated them on the 39,759 test orders (`01_exploration.ipynb` Cell 5):

| Metric | Clean Model ($T_0$ Compliant) | Leaky Model (Contaminated) | Deceptive Absolute Gap | Live Production Impact |
| :--- | :---: | :---: | :---: | :--- |
| **Capture Rate @ 10% Quota** | **21.20%** | **45.51%** | **+24.31%** | Clean captures pre-transit signals; Leaky crashes on NULL features in production. |
| **Precision @ 10% Quota** | **28.77%** | **61.77%** | **+33.00%** | Operations reaches truly dissatisfied buyers vs. contacting zero customers. |
| **Operational Lift @ 10%** | **2.12x** | **4.55x** | **+2.43x** | True 2.1x operational efficiency over blind random selection. |
| **ROC-AUC** | **0.5928** | **0.7753** | **+0.1825** | Artificially inflated ranking metric caused by future transit knowledge. |
| **PR-AUC (Average Precision)**| **0.2320** | **0.5152** | **+0.2832** | Naively suggests 51% precision; completely collapses in live deployment. |

### 5.2 Three Tempting Forbidden Columns Discarded
1. `order_delivered_customer_date`: Timestamp of delivery to buyer. Occurs 5 to 30 days after $T_0$.
2. `order_delivered_carrier_date`: Timestamp of hand-off to carrier. Occurs 1 to 5 days after $T_0$.
3. `order_status = 'delivered'`: Terminal status indicator revealing parcel fulfillment survival.

---

## 6. Feature Space Engineering & Behavioral Hypotheses

The table below documents all 21 pre-$T_0$ features alongside their behavioral hypotheses:

| Feature Name | Database Origin Table | Operational Behavioral Hypothesis Tested |
| :--- | :--- | :--- |
| `freight_ratio` | `olist.order_items` | High shipping costs relative to item price induce buyer grievance and amplify delay sensitivity. |
| `approval_delay_hours` | `olist.orders` | Extended payment authorization delay signals early friction and depletes customer patience. |
| `promised_delivery_days` | `olist.orders` | Lengthy platform delivery windows increase buyer anxiety and cancellation propensity. |
| `total_order_value` | `olist.order_payments` | High-value purchases create heightened customer scrutiny regarding packaging and speed. |
| `max_installments` | `olist.order_payments` | Many installments reflect credit-sensitive buyers with rigorous expectations of delivery timing. |
| `payment_types_count` | `olist.order_payments` | Split payments (vouchers + credit card) indicate multi-step billing and elevated dispute probability. |
| `has_boleto_payment` | `olist.order_payments` | Boleto payments correlate with traditional buyers experiencing offline processing delays. |
| `has_voucher_payment` | `olist.order_payments` | Promotional discount users exhibit higher impatience during peak campaign periods. |
| `total_items_count` | `olist.order_items` | Multi-item baskets increase warehouse picking complexity, missing item risk, and transit damage. |
| `unique_products_count` | `olist.order_items` | Product diversity within a basket demands complex staging, delaying merchant dispatch. |
| `unique_sellers_count` | `olist.order_items` | Multi-seller orders arrive in separate parcels at different times, creating buyer confusion. |
| `total_items_price` | `olist.order_items` | Basket item value establishes buyer perception of product tier and post-purchase service expectations. |
| `total_freight_value` | `olist.order_items` | Nominal freight charge directly impacts perceived shipping cost fairness. |
| `total_weight_g` | `olist.products` | Heavy parcels suffer manual sorting friction in postal hubs and elevated transit impact damage. |
| `total_volume_liters` | `olist.products` | Bulky parcels congest delivery vans and sit longer in regional intermediate cross-docks. |
| `avg_photos_qty` | `olist.products` | Sparse product catalog photography increases buyer expectation mismatch upon unboxing. |
| `avg_desc_length` | `olist.products` | Brief catalog descriptions indicate poor merchant documentation and higher return risk. |
| `is_interstate` | `olist.customers`, `olist.sellers` | Cross-border shipments undergo state tax inspection checkpoints (ICMS) and long transit legs. |
| `days_to_next_holiday` | BrasilAPI Calendar | Proximity to statutory national holidays causes postal hub congestion and buyer urgency. |
| `is_holiday_eve` | BrasilAPI Calendar | Orders confirmed within 3 days of major holidays face immediate logistics network shutdowns. |
| `is_black_friday_week` | BrasilAPI Calendar | Late-November promotional orders experience systemic backlogs across Brazilian freight carriers. |

### 6.1 Baseline Comparison with Simple Heuristic Rules (No-Model Benchmarks)
To test whether a machine learning model is necessary, we benchmarked simple operational heuristic rules against our models (`02_system.ipynb` Cell 4):

| Prioritization Approach | Operational Decision Rule | Capture Rate @ 10% | Precision @ 10% | Lift @ 10% |
| :--- | :--- | :---: | :---: | :---: |
| **Random Baseline** | Proactively contact random 10% of daily orders | 10.00% | 13.57% | 1.00x |
| **Simple Heuristic 1** | Contact top 10% orders with highest `freight_ratio` | 10.78% | 14.64% | 1.08x |
| **Simple Heuristic 2** | Contact top 10% orders with longest `promised_delivery_days` | 10.69% | 14.51% | 1.07x |
| **Baseline Linear Model** | Regularized Logistic Regression with class weighting | 20.40% | 27.69% | 2.04x |
| **Production LightGBM** | Tuned Gradient Boosted Trees with non-linear interactions | **21.20%** | **28.77%** | **2.12x** |

*Operational Takeaway:* Heuristic rules achieve only a 0.78% improvement over random calling. Our production LightGBM model **more than doubles follow-up efficiency (2.12x lift)**.

---

## 7. Capacity Portfolio Evaluation & August 2018 Stability

### 7.1 Capacity Portfolio Benchmarks (5%, 10%, 20%)
Support leadership can adjust contact quotas based on workforce availability (`02_system.ipynb` Cell 4):

| Operational Capacity Scenario | Contacted Orders | Negative Review Capture Rate | Precision | Operational Lift |
| :---: | :---: | :---: | :---: | :---: |
| **Constrained (5% Quota)** | 1,988 orders | 11.53% | 31.31% | 2.31x |
| **Standard (10% Quota)** | 3,976 orders | **21.20%** | **28.77%** | **2.12x** |
| **Expanded (20% Quota)** | 7,952 orders | 38.30% | 25.99% | 1.91x |

### 7.2 Standalone Performance on Final Month (August 2018)
On the final holdout month of August 2018 (6,368 confirmed orders):
* **Capture Rate @ 10%:** 21.01%
* **Precision @ 10%:** 22.36%
* **Operational Lift:** 2.10x

The close alignment between August 2018 (21.01%) and the overall test set (21.20%) confirms that the model generalizes robustly without temporal overfitting.

---

## 8. Distribution Drift Analysis & 6-Month Production Degradation Forecast

### 8.1 Observed Train vs. Test Distribution Shifts
Comparing orders before March 1, 2018 with orders from March to August 2018 (`02_system.ipynb` Cell 6):
* `days_to_next_holiday`: Mean shifted from 23.5 days to 32.0 days (+35.9%) due to fewer statutory holidays in Brazilian autumn/winter.
* `promised_delivery_days`: Mean dropped from 24.6 to 21.4 days (-13.0%) as Olist algorithms tightened delivery commitments.
* `is_black_friday_week`: Shifted from 7.1% in train to 0% in test (Black Friday occurs only in November).

Despite these seasonal and platform shifts, model operational lift remained remarkably stable between **1.72x and 2.62x** across all 6 test months.

### 8.2 Six-Month Production Degradation Forecast
In a live production deployment:
1. **What breaks first?** Courier freight rate tables and platform SLA promise rules (`freight_ratio` and `promised_delivery_days`). Carrier contracts undergo semi-annual tariff revisions, drifting the calibrated probability distributions.
2. **How do we detect it?** Weekly tracking of Population Stability Index (PSI) on all 21 pre-$T_0$ features; an automated alert triggers retraining if $\text{PSI} > 0.2$ on core logistics features, or if the 90th percentile predicted probability drifts by more than $\pm 0.05$.

---

## 9. Engineering Retrospective & AI Tool Usage Note

Delivering this Capstone required synthesizing techniques across all three previous challenges:
* **Timezone Parsing Trap:** We encountered silent day-difference calculation bugs where naive timestamps interacted with São Paulo local offsets (`America/Sao_Paulo`), resolved by strictly normalizing all database timestamps to UTC.
* **Execution Budget:** Running the entire master pipeline, from PostgreSQL view deployment and test execution to MLflow logging and dual notebook execution, runs in **42.88 seconds**, well within the 45-minute capstone SLA.

### Candid Note on AI Tool Usage
During project development, AI tools were utilized strictly as code accelerators for boilerplate engineering tasks: drafting initial unit test method templates, generating initial regex patterns for forbidden column interception, and formatting repetitive markdown tables. 

All primary architectural decisions—including defining the strict decision boundary at $T_0$, empirically testing and rejecting weather and inflation APIs, designing PostgreSQL materialized views and composite B-tree indexes, formulating the 10% operational capacity mathematics, structuring parent-child time-series MLflow runs, and diagnosing the deceptive leakage gap—were conceived, calculated, implemented, and verified directly by Shayan Ganji.
