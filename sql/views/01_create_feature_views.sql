-- ============================================================================
-- Early Warning System: In-Database Feature Engineering (T0 Decision Moment)
-- Schema: olist
-- Target SLA: Execution Latency < 500 ms
-- ============================================================================

-- 1. Create standard dynamic view for live inference and scoring
CREATE OR REPLACE VIEW olist.view_order_early_warning_features AS
WITH payment_agg AS (
    SELECT 
        order_id,
        SUM(payment_value) AS total_order_value,
        MAX(payment_installments) AS max_installments,
        COUNT(DISTINCT payment_type) AS payment_types_count,
        MAX(CASE WHEN payment_type = 'boleto' THEN 1 ELSE 0 END) AS has_boleto_payment,
        MAX(CASE WHEN payment_type = 'voucher' THEN 1 ELSE 0 END) AS has_voucher_payment
    FROM olist.order_payments
    GROUP BY order_id
),
items_agg AS (
    SELECT 
        oi.order_id,
        COUNT(oi.order_item_id) AS total_items_count,
        COUNT(DISTINCT oi.product_id) AS unique_products_count,
        COUNT(DISTINCT oi.seller_id) AS unique_sellers_count,
        SUM(oi.price) AS total_items_price,
        SUM(oi.freight_value) AS total_freight_value,
        COALESCE(SUM(p.product_weight_g), 0) AS total_weight_g,
        COALESCE(SUM(p.product_length_cm * p.product_height_cm * p.product_width_cm) / 1000.0, 0) AS total_volume_liters,
        COALESCE(AVG(p.product_photos_qty), 0) AS avg_photos_qty,
        COALESCE(AVG(p.product_description_lenght), 0) AS avg_desc_length,
        MAX(CASE WHEN c.customer_state != s.seller_state THEN 1 ELSE 0 END) AS is_interstate,
        MAX(s.seller_state) AS dominant_seller_state
    FROM olist.order_items oi
    JOIN olist.orders o ON oi.order_id = o.order_id
    JOIN olist.customers c ON o.customer_id = c.customer_id
    JOIN olist.sellers s ON oi.seller_id = s.seller_id
    LEFT JOIN olist.products p ON oi.product_id = p.product_id
    GROUP BY oi.order_id
),
reviews_agg AS (
    SELECT 
        order_id, 
        MIN(review_score) AS review_score
    FROM olist.order_reviews
    GROUP BY order_id
)
SELECT 
    o.order_id,
    o.customer_id,
    o.order_purchase_timestamp,
    o.order_approved_at,
    c.customer_state,
    ROUND(EXTRACT(EPOCH FROM (o.order_approved_at - o.order_purchase_timestamp)) / 3600.0, 2) AS approval_delay_hours,
    ROUND(EXTRACT(EPOCH FROM (o.order_estimated_delivery_date - o.order_approved_at)) / 86400.0, 2) AS promised_delivery_days,
    COALESCE(pa.total_order_value, 0) AS total_order_value,
    COALESCE(pa.max_installments, 1) AS max_installments,
    COALESCE(pa.payment_types_count, 1) AS payment_types_count,
    COALESCE(pa.has_boleto_payment, 0) AS has_boleto_payment,
    COALESCE(pa.has_voucher_payment, 0) AS has_voucher_payment,
    COALESCE(ia.total_items_count, 0) AS total_items_count,
    COALESCE(ia.unique_products_count, 0) AS unique_products_count,
    COALESCE(ia.unique_sellers_count, 0) AS unique_sellers_count,
    COALESCE(ia.total_items_price, 0) AS total_items_price,
    COALESCE(ia.total_freight_value, 0) AS total_freight_value,
    ROUND(COALESCE(ia.total_freight_value, 0) / NULLIF(COALESCE(ia.total_items_price, 0), 0), 4) AS freight_ratio,
    COALESCE(ia.total_weight_g, 0) AS total_weight_g,
    COALESCE(ia.total_volume_liters, 0) AS total_volume_liters,
    ROUND(COALESCE(ia.avg_photos_qty, 0), 2) AS avg_photos_qty,
    ROUND(COALESCE(ia.avg_desc_length, 0), 2) AS avg_desc_length,
    COALESCE(ia.is_interstate, 0) AS is_interstate,
    ia.dominant_seller_state,
    r.review_score,
    CASE 
        WHEN r.review_score IN (1, 2) THEN 1 
        WHEN r.review_score IS NOT NULL THEN 0 
        ELSE NULL 
    END AS is_negative_review
FROM olist.orders o
JOIN olist.customers c ON o.customer_id = c.customer_id
LEFT JOIN payment_agg pa ON o.order_id = pa.order_id
LEFT JOIN items_agg ia ON o.order_id = ia.order_id
LEFT JOIN reviews_agg r ON o.order_id = r.order_id
WHERE o.order_approved_at IS NOT NULL;

-- 2. Create materialized view for ultra-low latency model training and backtesting
DROP MATERIALIZED VIEW IF EXISTS olist.mv_order_early_warning_features CASCADE;

CREATE MATERIALIZED VIEW olist.mv_order_early_warning_features AS
SELECT * FROM olist.view_order_early_warning_features;

-- 3. Create high-performance B-Tree indexes on the materialized view
CREATE UNIQUE INDEX idx_mv_early_warning_order_id 
    ON olist.mv_order_early_warning_features (order_id);

CREATE INDEX idx_mv_early_warning_approved_at 
    ON olist.mv_order_early_warning_features (order_approved_at);

CREATE INDEX idx_mv_early_warning_customer_state 
    ON olist.mv_order_early_warning_features (customer_state);

