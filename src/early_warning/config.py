"""
Configuration module for Early Warning Decision Engine.
Centralizes database connection, file paths, external API URLs, and hyperparameters.
"""

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class SystemConfig:
    """Immutable system configuration parameters."""

    # Project paths
    project_root: Path = Path(__file__).resolve().parent.parent.parent
    data_dir: Path = project_root / "data"
    raw_data_dir: Path = data_dir / "raw"
    external_data_dir: Path = data_dir / "external"

    # Database settings
    db_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://postgres@127.0.0.1:5432/olist_db"
    )

    # Operational constraints
    target_capacity_ratio: float = 0.10  # 10% operational follow-up quota
    alternative_capacity_ratios: tuple = (0.05, 0.10, 0.20)

    # Temporal split cutoff date (YYYY-MM-DD)
    # Orders confirmed before this date are Train, on/after are Evaluation (Test)
    train_test_split_date: str = "2018-03-01"

    # Random seed for absolute reproducibility
    random_seed: int = 42

    # External API configuration
    brasil_api_holiday_url: str = "https://brasilapi.com.br/api/feriados/v1"
    open_meteo_archive_url: str = "https://archive-api.open-meteo.com/v1/archive"

    @property
    def db_connection_url(self) -> str:
        return self.db_url


DEFAULT_CONFIG = SystemConfig()
default_config = DEFAULT_CONFIG
