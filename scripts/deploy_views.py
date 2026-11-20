"""Database Feature View deployment and EXPLAIN plan generator."""

import psycopg2


def deploy_views_and_capture_plans():
    conn = psycopg2.connect(
        dbname="olist_db",
        user="postgres",
        host="127.0.0.1",
        port=5432
    )
    conn.autocommit = True
    cur = conn.cursor()

    # Read SQL file
    with open("sql/views/01_create_feature_views.sql", "r", encoding="utf-8") as f:
        sql_commands = f.read()

    print("Deploying views to PostgreSQL olist_db...")
    cur.execute(sql_commands)
    print("Views and indexes successfully created!")

    # Capture Plan 1: Single Order Point Lookup from View (Online Inference SLA)
    order_id_sample = "e481f51cbdc54678b7cc49136f2d6af7"
    cur.execute(f"""
    EXPLAIN (ANALYZE, BUFFERS)
    SELECT * FROM olist.view_order_early_warning_features
    WHERE order_id = '{order_id_sample}';
    """)
    plan_single = cur.fetchall()

    # Capture Plan 2: Batch Window Scan from Materialized View (Evaluation & Backtesting SLA)
    cur.execute("""
    EXPLAIN (ANALYZE, BUFFERS)
    SELECT * FROM olist.mv_order_early_warning_features
    WHERE order_approved_at >= '2018-08-01' AND order_approved_at < '2018-09-01';
    """)
    plan_batch = cur.fetchall()

    # Capture Plan 3: Full Dataset Training Scan
    cur.execute("""
    EXPLAIN (ANALYZE, BUFFERS)
    SELECT * FROM olist.mv_order_early_warning_features;
    """)
    plan_full = cur.fetchall()

    # Write plans to documentation file
    output_lines = [
        "=" * 80,
        "POSTGRESQL FEATURE EXTRACTION QUERY EXECUTION PLANS",
        "Target SLA: Latency < 500 ms",
        "=" * 80,
        "\n--- PLAN 1: Point Lookup for Live Single-Order Scoring (View) ---",
        f"Query: SELECT * FROM olist.view_order_early_warning_features WHERE order_id = '{order_id_sample}';",
        "-" * 80,
    ]
    output_lines.extend(row[0] for row in plan_single)

    aug_query = (
        "SELECT * FROM olist.mv_order_early_warning_features "
        "WHERE order_approved_at >= '2018-08-01' AND order_approved_at < '2018-09-01';"
    )
    output_lines.extend([
        "\n" + "=" * 80,
        "--- PLAN 2: Monthly Cohort Filter (August 2018 Final Evaluation Set) ---",
        f"Query: {aug_query}",
        "-" * 80,
    ])
    output_lines.extend(row[0] for row in plan_batch)

    output_lines.extend([
        "\n" + "=" * 80,
        "--- PLAN 3: Full Table Scan for Training Dataset (99k orders) ---",
        "Query: SELECT * FROM olist.mv_order_early_warning_features;",
        "-" * 80,
    ])
    output_lines.extend(row[0] for row in plan_full)

    with open("sql/views/feature_query_plan.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines) + "\n")

    print("Execution plans written to sql/views/feature_query_plan.txt")

    # Print summary execution times
    print("\nSummary Execution Times:")
    for plan, label in [
        (plan_single, "Plan 1 (Point Lookup)"),
        (plan_batch, "Plan 2 (August 2018 Cohort)"),
        (plan_full, "Plan 3 (Full Scan 99k)")
    ]:
        for line in plan:
            if "Execution Time:" in line[0]:
                print(f"  - {label}: {line[0]}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    deploy_views_and_capture_plans()
