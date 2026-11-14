"""Temporal Audit and Zero-Leakage Enforcement Module.

This module defines the lifecycle availability of every column across the Olist
relational schema relative to the decision moment T0 (order_approved_at).
It provides a runtime guard that verifies feature matrices are strictly
free of post-T0 information, preventing target and temporal leakage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Optional, Set

import pandas as pd


class ColumnLifecycleStatus(str, Enum):
    """Classification of schema columns based on their temporal lifecycle."""

    ALLOWED_AT_T0 = "ALLOWED_AT_T0"
    FORBIDDEN_POST_T0 = "FORBIDDEN_POST_T0"
    TARGET_ONLY = "TARGET_ONLY"
    METADATA_IDENTIFIER = "METADATA_IDENTIFIER"


class TemporalLeakageError(ValueError):
    """Raised when a feature matrix, query, or model pipeline consumes
    attributes or derived features originating after decision timestamp T0.
    """


@dataclass(frozen=True)
class ColumnAuditRecord:
    """Detailed audit metadata for an individual database column."""

    table_name: str
    column_name: str
    status: ColumnLifecycleStatus
    lifecycle_event: str
    rationale: str


# Comprehensive schema audit covering all 8 tables in olist_db
SCHEMA_AUDIT_REGISTRY: List[ColumnAuditRecord] = [
    # Table: olist.orders
    ColumnAuditRecord(
        table_name="olist.orders",
        column_name="order_id",
        status=ColumnLifecycleStatus.METADATA_IDENTIFIER,
        lifecycle_event="Checkout",
        rationale="Primary key identifier of the order. Permissible for joins and index tracking.",
    ),
    ColumnAuditRecord(
        table_name="olist.orders",
        column_name="customer_id",
        status=ColumnLifecycleStatus.METADATA_IDENTIFIER,
        lifecycle_event="Checkout",
        rationale="Order-specific customer token. Permissible for joining customer location.",
    ),
    ColumnAuditRecord(
        table_name="olist.orders",
        column_name="order_status",
        status=ColumnLifecycleStatus.FORBIDDEN_POST_T0,
        lifecycle_event="Order Lifecycle",
        rationale=(
            "Terminal status ('delivered', 'canceled', 'shipped') reveals future order survival. "
            "At T0, every approved order is strictly 'approved' or 'invoiced'."
        ),
    ),
    ColumnAuditRecord(
        table_name="olist.orders",
        column_name="order_purchase_timestamp",
        status=ColumnLifecycleStatus.ALLOWED_AT_T0,
        lifecycle_event="Checkout",
        rationale="Exact moment customer pressed purchase button. Strictly precedes or equals T0.",
    ),
    ColumnAuditRecord(
        table_name="olist.orders",
        column_name="order_approved_at",
        status=ColumnLifecycleStatus.ALLOWED_AT_T0,
        lifecycle_event="Order Approval (T0)",
        rationale="The definitive decision boundary T0. Order payment is settled and fulfillment begins.",
    ),
    ColumnAuditRecord(
        table_name="olist.orders",
        column_name="order_delivered_carrier_date",
        status=ColumnLifecycleStatus.FORBIDDEN_POST_T0,
        lifecycle_event="Carrier Hand-off",
        rationale="Occurs 1 to 5 days after T0 when the seller drops the parcel at the postal hub.",
    ),
    ColumnAuditRecord(
        table_name="olist.orders",
        column_name="order_delivered_customer_date",
        status=ColumnLifecycleStatus.FORBIDDEN_POST_T0,
        lifecycle_event="Customer Delivery",
        rationale="Occurs 5 to 30 days after T0. Prime source of target leakage in naive systems.",
    ),
    ColumnAuditRecord(
        table_name="olist.orders",
        column_name="order_estimated_delivery_date",
        status=ColumnLifecycleStatus.ALLOWED_AT_T0,
        lifecycle_event="Checkout",
        rationale=(
            "Initial promise date given to the customer at checkout. Allowed only to compute promised "
            "delivery duration at T0. Cannot be compared with actual delivery date."
        ),
    ),
    # Table: olist.order_items
    ColumnAuditRecord(
        table_name="olist.order_items",
        column_name="order_id",
        status=ColumnLifecycleStatus.METADATA_IDENTIFIER,
        lifecycle_event="Checkout",
        rationale="Foreign key referencing olist.orders.",
    ),
    ColumnAuditRecord(
        table_name="olist.order_items",
        column_name="order_item_id",
        status=ColumnLifecycleStatus.ALLOWED_AT_T0,
        lifecycle_event="Checkout",
        rationale="Sequential item index. Used to compute total basket item count at T0.",
    ),
    ColumnAuditRecord(
        table_name="olist.order_items",
        column_name="product_id",
        status=ColumnLifecycleStatus.ALLOWED_AT_T0,
        lifecycle_event="Checkout",
        rationale="Product identifier. Available at T0 to join catalog attributes.",
    ),
    ColumnAuditRecord(
        table_name="olist.order_items",
        column_name="seller_id",
        status=ColumnLifecycleStatus.ALLOWED_AT_T0,
        lifecycle_event="Checkout",
        rationale="Seller identifier. Available at T0 to join seller state and geography.",
    ),
    ColumnAuditRecord(
        table_name="olist.order_items",
        column_name="shipping_limit_date",
        status=ColumnLifecycleStatus.ALLOWED_AT_T0,
        lifecycle_event="Checkout",
        rationale="SLA deadline assigned to the seller at checkout for dispatching goods.",
    ),
    ColumnAuditRecord(
        table_name="olist.order_items",
        column_name="price",
        status=ColumnLifecycleStatus.ALLOWED_AT_T0,
        lifecycle_event="Checkout",
        rationale="Item unit price locked in at transaction approval.",
    ),
    ColumnAuditRecord(
        table_name="olist.order_items",
        column_name="freight_value",
        status=ColumnLifecycleStatus.ALLOWED_AT_T0,
        lifecycle_event="Checkout",
        rationale="Shipping fee charged to the customer at checkout.",
    ),
    # Table: olist.order_payments
    ColumnAuditRecord(
        table_name="olist.order_payments",
        column_name="order_id",
        status=ColumnLifecycleStatus.METADATA_IDENTIFIER,
        lifecycle_event="Checkout",
        rationale="Foreign key referencing olist.orders.",
    ),
    ColumnAuditRecord(
        table_name="olist.order_payments",
        column_name="payment_sequential",
        status=ColumnLifecycleStatus.ALLOWED_AT_T0,
        lifecycle_event="Checkout",
        rationale="Sequence index of payment tenders.",
    ),
    ColumnAuditRecord(
        table_name="olist.order_payments",
        column_name="payment_type",
        status=ColumnLifecycleStatus.ALLOWED_AT_T0,
        lifecycle_event="Payment Authorization",
        rationale="Payment instrument (credit_card, boleto, voucher, debit_card). Available at T0.",
    ),
    ColumnAuditRecord(
        table_name="olist.order_payments",
        column_name="payment_installments",
        status=ColumnLifecycleStatus.ALLOWED_AT_T0,
        lifecycle_event="Payment Authorization",
        rationale="Number of installment months chosen by buyer. Available at T0.",
    ),
    ColumnAuditRecord(
        table_name="olist.order_payments",
        column_name="payment_value",
        status=ColumnLifecycleStatus.ALLOWED_AT_T0,
        lifecycle_event="Payment Authorization",
        rationale="Monetary amount charged per tender. Available at T0.",
    ),
    # Table: olist.order_reviews
    ColumnAuditRecord(
        table_name="olist.order_reviews",
        column_name="review_id",
        status=ColumnLifecycleStatus.TARGET_ONLY,
        lifecycle_event="Post-Delivery Survey",
        rationale="Unique survey response ID. Forbidden as a feature.",
    ),
    ColumnAuditRecord(
        table_name="olist.order_reviews",
        column_name="order_id",
        status=ColumnLifecycleStatus.METADATA_IDENTIFIER,
        lifecycle_event="Post-Delivery Survey",
        rationale="Foreign key referencing olist.orders.",
    ),
    ColumnAuditRecord(
        table_name="olist.order_reviews",
        column_name="review_score",
        status=ColumnLifecycleStatus.TARGET_ONLY,
        lifecycle_event="Post-Delivery Survey",
        rationale=(
            "Customer satisfaction score (1 to 5). Generates target variable (y = 1 if score in [1, 2]). "
            "Strictly forbidden from entering the feature space."
        ),
    ),
    ColumnAuditRecord(
        table_name="olist.order_reviews",
        column_name="review_comment_title",
        status=ColumnLifecycleStatus.FORBIDDEN_POST_T0,
        lifecycle_event="Post-Delivery Survey",
        rationale="Customer review title written weeks after delivery. Major leakage.",
    ),
    ColumnAuditRecord(
        table_name="olist.order_reviews",
        column_name="review_comment_message",
        status=ColumnLifecycleStatus.FORBIDDEN_POST_T0,
        lifecycle_event="Post-Delivery Survey",
        rationale="Customer review text written weeks after delivery. Major leakage.",
    ),
    ColumnAuditRecord(
        table_name="olist.order_reviews",
        column_name="review_creation_date",
        status=ColumnLifecycleStatus.FORBIDDEN_POST_T0,
        lifecycle_event="Post-Delivery Survey",
        rationale="Date survey link was generated for customer. Occurs after delivery.",
    ),
    ColumnAuditRecord(
        table_name="olist.order_reviews",
        column_name="review_answer_timestamp",
        status=ColumnLifecycleStatus.FORBIDDEN_POST_T0,
        lifecycle_event="Post-Delivery Survey",
        rationale="Timestamp customer submitted the review. Occurs days/weeks after T0.",
    ),
    # Table: olist.products
    ColumnAuditRecord(
        table_name="olist.products",
        column_name="product_id",
        status=ColumnLifecycleStatus.METADATA_IDENTIFIER,
        lifecycle_event="Catalog Ingestion",
        rationale="Product identifier.",
    ),
    ColumnAuditRecord(
        table_name="olist.products",
        column_name="product_category_name",
        status=ColumnLifecycleStatus.ALLOWED_AT_T0,
        lifecycle_event="Catalog Ingestion",
        rationale="Static category taxonomy. Available at T0.",
    ),
    ColumnAuditRecord(
        table_name="olist.products",
        column_name="product_name_lenght",
        status=ColumnLifecycleStatus.ALLOWED_AT_T0,
        lifecycle_event="Catalog Ingestion",
        rationale="Listing title length. Available at T0.",
    ),
    ColumnAuditRecord(
        table_name="olist.products",
        column_name="product_description_lenght",
        status=ColumnLifecycleStatus.ALLOWED_AT_T0,
        lifecycle_event="Catalog Ingestion",
        rationale="Listing description length. Available at T0.",
    ),
    ColumnAuditRecord(
        table_name="olist.products",
        column_name="product_photos_qty",
        status=ColumnLifecycleStatus.ALLOWED_AT_T0,
        lifecycle_event="Catalog Ingestion",
        rationale="Listing photo count. Available at T0.",
    ),
    ColumnAuditRecord(
        table_name="olist.products",
        column_name="product_weight_g",
        status=ColumnLifecycleStatus.ALLOWED_AT_T0,
        lifecycle_event="Catalog Ingestion",
        rationale="Item physical shipping weight. Available at T0.",
    ),
    ColumnAuditRecord(
        table_name="olist.products",
        column_name="product_length_cm",
        status=ColumnLifecycleStatus.ALLOWED_AT_T0,
        lifecycle_event="Catalog Ingestion",
        rationale="Item physical dimensions. Available at T0.",
    ),
    ColumnAuditRecord(
        table_name="olist.products",
        column_name="product_height_cm",
        status=ColumnLifecycleStatus.ALLOWED_AT_T0,
        lifecycle_event="Catalog Ingestion",
        rationale="Item physical dimensions. Available at T0.",
    ),
    ColumnAuditRecord(
        table_name="olist.products",
        column_name="product_width_cm",
        status=ColumnLifecycleStatus.ALLOWED_AT_T0,
        lifecycle_event="Catalog Ingestion",
        rationale="Item physical dimensions. Available at T0.",
    ),
    # Table: olist.sellers
    ColumnAuditRecord(
        table_name="olist.sellers",
        column_name="seller_id",
        status=ColumnLifecycleStatus.METADATA_IDENTIFIER,
        lifecycle_event="Seller Onboarding",
        rationale="Seller identifier.",
    ),
    ColumnAuditRecord(
        table_name="olist.sellers",
        column_name="seller_zip_code_prefix",
        status=ColumnLifecycleStatus.ALLOWED_AT_T0,
        lifecycle_event="Seller Onboarding",
        rationale="Seller postal code prefix. Available at T0.",
    ),
    ColumnAuditRecord(
        table_name="olist.sellers",
        column_name="seller_city",
        status=ColumnLifecycleStatus.ALLOWED_AT_T0,
        lifecycle_event="Seller Onboarding",
        rationale="Seller municipality. Available at T0.",
    ),
    ColumnAuditRecord(
        table_name="olist.sellers",
        column_name="seller_state",
        status=ColumnLifecycleStatus.ALLOWED_AT_T0,
        lifecycle_event="Seller Onboarding",
        rationale="Seller federative unit (state). Available at T0.",
    ),
    # Table: olist.customers
    ColumnAuditRecord(
        table_name="olist.customers",
        column_name="customer_id",
        status=ColumnLifecycleStatus.METADATA_IDENTIFIER,
        lifecycle_event="Customer Registration",
        rationale="Order-level customer key.",
    ),
    ColumnAuditRecord(
        table_name="olist.customers",
        column_name="customer_unique_id",
        status=ColumnLifecycleStatus.METADATA_IDENTIFIER,
        lifecycle_event="Customer Registration",
        rationale="Persistent customer identity. Available at T0.",
    ),
    ColumnAuditRecord(
        table_name="olist.customers",
        column_name="customer_zip_code_prefix",
        status=ColumnLifecycleStatus.ALLOWED_AT_T0,
        lifecycle_event="Customer Registration",
        rationale="Customer postal code prefix. Available at T0.",
    ),
    ColumnAuditRecord(
        table_name="olist.customers",
        column_name="customer_city",
        status=ColumnLifecycleStatus.ALLOWED_AT_T0,
        lifecycle_event="Customer Registration",
        rationale="Customer municipality. Available at T0.",
    ),
    ColumnAuditRecord(
        table_name="olist.customers",
        column_name="customer_state",
        status=ColumnLifecycleStatus.ALLOWED_AT_T0,
        lifecycle_event="Customer Registration",
        rationale="Customer federative unit (state). Available at T0.",
    ),
]

# Patterns in feature column names that signify post-T0 temporal contamination
FORBIDDEN_LEAKAGE_REGEX_PATTERNS: List[re.Pattern] = [
    re.compile(r"deliver.*date", re.IGNORECASE),
    re.compile(r"actual.*deliver", re.IGNORECASE),
    re.compile(r"carrier.*date", re.IGNORECASE),
    re.compile(r"delay.*day", re.IGNORECASE),
    re.compile(r"is.*delay", re.IGNORECASE),
    re.compile(r"review.*score", re.IGNORECASE),
    re.compile(r"review.*message", re.IGNORECASE),
    re.compile(r"review.*comment", re.IGNORECASE),
    re.compile(r"review.*date", re.IGNORECASE),
    re.compile(r"review.*answer", re.IGNORECASE),
    re.compile(r"^target$", re.IGNORECASE),
    re.compile(r"is_negative_review", re.IGNORECASE),
]

# Explicit forbidden column names
FORBIDDEN_EXACT_COLUMN_NAMES: Set[str] = {
    record.column_name
    for record in SCHEMA_AUDIT_REGISTRY
    if record.status in (ColumnLifecycleStatus.FORBIDDEN_POST_T0, ColumnLifecycleStatus.TARGET_ONLY)
} | {
    "actual_delivery_days",
    "delivery_delay_days",
    "is_delayed",
    "is_late",
    "carrier_delay",
    "carrier_transit_days",
    "target",
    "is_negative_review",
}


def get_audit_dataframe() -> pd.DataFrame:
    """Return the temporal lifecycle audit table as a pandas DataFrame."""
    records = [
        {
            "table_name": rec.table_name,
            "column_name": rec.column_name,
            "status": rec.status.value,
            "lifecycle_event": rec.lifecycle_event,
            "rationale": rec.rationale,
        }
        for rec in SCHEMA_AUDIT_REGISTRY
    ]
    return pd.DataFrame(records)


def is_column_allowed_at_t0(column_name: str) -> bool:
    """Check whether a column name is permissible at decision moment T0."""
    normalized = column_name.strip().lower()
    if normalized in FORBIDDEN_EXACT_COLUMN_NAMES:
        return False
    for pattern in FORBIDDEN_LEAKAGE_REGEX_PATTERNS:
        if pattern.search(normalized):
            return False
    return True


def verify_zero_leakage(df: pd.DataFrame, allowed_whitelist: Optional[Iterable[str]] = None) -> None:
    """Assert that a feature matrix contains zero post-T0 or target leakage columns.

    Args:
        df: The pandas DataFrame representing the feature matrix or inference input.
        allowed_whitelist: Optional set of strictly authorized feature names.

    Raises:
        TemporalLeakageError: If any forbidden column or regex pattern is detected.
    """
    detected_violations: List[str] = []

    whitelist_set = {col.lower() for col in allowed_whitelist} if allowed_whitelist else None

    for col in df.columns:
        col_lower = col.strip().lower()

        # If a strict whitelist is provided, ensure col belongs to whitelist
        if whitelist_set is not None and col_lower not in whitelist_set:
            detected_violations.append(
                f"Column '{col}' is not present in the authorized feature whitelist."
            )
            continue

        # Check exact forbidden names
        if col_lower in FORBIDDEN_EXACT_COLUMN_NAMES:
            detected_violations.append(
                f"Column '{col}' matches forbidden post-T0 or target column name."
            )
            continue

        # Check regex leakage patterns
        for pattern in FORBIDDEN_LEAKAGE_REGEX_PATTERNS:
            if pattern.search(col_lower):
                detected_violations.append(
                    f"Column '{col}' matches temporal leakage pattern '{pattern.pattern}'."
                )
                break

    if detected_violations:
        error_msg = (
            f"TEMPORAL LEAKAGE DETECTED: {len(detected_violations)} violation(s) found:\n"
            + "\n".join(f"  - {v}" for v in detected_violations)
            + "\nImmediate abort: Pipeline must only consume attributes available at or before T0 (order_approved_at)."
        )
        raise TemporalLeakageError(error_msg)
