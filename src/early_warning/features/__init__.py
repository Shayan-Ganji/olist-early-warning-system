"""Feature engineering and temporal audit package for Early Warning System."""

from early_warning.features.sql_features import (
    AUTHORIZED_FEATURE_COLUMNS,
    METADATA_COLUMNS,
    FeatureDataLoader,
)
from early_warning.features.temporal_audit import (
    ColumnAuditRecord,
    ColumnLifecycleStatus,
    SCHEMA_AUDIT_REGISTRY,
    TemporalLeakageError,
    get_audit_dataframe,
    is_column_allowed_at_t0,
    verify_zero_leakage,
)

__all__ = [
    "AUTHORIZED_FEATURE_COLUMNS",
    "METADATA_COLUMNS",
    "FeatureDataLoader",
    "ColumnAuditRecord",
    "ColumnLifecycleStatus",
    "SCHEMA_AUDIT_REGISTRY",
    "TemporalLeakageError",
    "get_audit_dataframe",
    "is_column_allowed_at_t0",
    "verify_zero_leakage",
]
