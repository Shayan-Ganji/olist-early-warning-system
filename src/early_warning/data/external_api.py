"""
External Data Ingestion and Caching Module.
Implements leak-free API client for Brazilian National Holidays (BrasilAPI)
and provides offline benchmark harness for rejected alternatives (Open-Meteo).
"""

from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import logging
import urllib.request
import pandas as pd
import numpy as np

from early_warning.config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class BrasilAPIHolidayClient:
    """
    Client for Brazilian National Holidays via BrasilAPI (https://brasilapi.com.br).
    
    Architectural Guarantees:
    - Zero Temporal Leakage: National statutory holidays are enacted by federal decree
      years in advance. At T_0 (order_approved_at), the calendar is 100% deterministic.
    - Zero Runtime Dependency: Caches API responses to local disk for 100% offline testing.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        base_url: str = DEFAULT_CONFIG.brasil_api_holiday_url
    ) -> None:
        self.cache_dir = cache_dir or DEFAULT_CONFIG.external_data_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "holidays_br.json"
        self.base_url = base_url
        self._holidays_cache: Optional[List[Dict[str, Any]]] = None

    def fetch_holidays(self, years: tuple[int, ...] = (2016, 2017, 2018), force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Fetch national holidays for specified years with local persistent caching.
        """
        if not force_refresh and self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self._holidays_cache = json.load(f)
                logger.info(f"Loaded {len(self._holidays_cache)} Brazilian holidays from local cache: {self.cache_file}")
                return self._holidays_cache
            except Exception as e:
                logger.warning(f"Cache read failed ({e}), re-fetching from BrasilAPI...")

        all_holidays: List[Dict[str, Any]] = []
        headers = {"User-Agent": "OlistEarlyWarning/1.0 (Academic Research; Dotin Lab Term 3)"}

        for yr in years:
            url = f"{self.base_url}/{yr}"
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status == 200:
                        yr_holidays = json.loads(resp.read().decode("utf-8"))
                        for h in yr_holidays:
                            h["year"] = yr
                            all_holidays.append(h)
                        logger.info(f"Retrieved {len(yr_holidays)} holidays for year {yr} from BrasilAPI.")
            except Exception as e:
                logger.error(f"Failed to fetch holidays for year {yr} from {url}: {e}")
                # Fallback static calendar for critical offline resilience
                all_holidays.extend(self._get_fallback_holidays(yr))

        # Persist to local disk cache
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(all_holidays, f, ensure_ascii=False, indent=2)

        self._holidays_cache = all_holidays
        return all_holidays

    def get_holiday_dates(self, years: tuple[int, ...] = (2016, 2017, 2018)) -> np.ndarray:
        """Return sorted array of holiday datetime.date objects."""
        holidays = self.fetch_holidays(years=years)
        dates = [datetime.strptime(h["date"], "%Y-%m-%d").date() for h in holidays]
        return np.array(sorted(set(dates)))

    def compute_holiday_features(self, approved_timestamps: pd.Series) -> pd.DataFrame:
        """
        Vectorized computation of calendar closeness features relative to T_0.
        
        Outputs:
        - days_to_next_holiday: Days remaining until the nearest future public holiday.
        - is_holiday_eve: Binary indicator (1 if next holiday is within 3 days, else 0).
        - is_black_friday_week: Binary indicator for intense promotional peak in late November.
        """
        holiday_dates = self.get_holiday_dates()
        
        # Convert timestamps to date series safely handling timezone
        dt_series = pd.to_datetime(approved_timestamps, utc=True).dt.tz_convert(None)
        order_dates = dt_series.dt.date

        days_to_next = []
        is_holiday_eve = []
        is_bf_week = []

        for ord_date in order_dates:
            if pd.isna(ord_date):
                days_to_next.append(np.nan)
                is_holiday_eve.append(0)
                is_bf_week.append(0)
                continue

            # Find future holidays on or after order confirmation (T_0)
            future_mask = holiday_dates >= ord_date
            if np.any(future_mask):
                next_h = holiday_dates[future_mask][0]
                delta_days = (next_h - ord_date).days
            else:
                delta_days = 99  # Far away if beyond calendar boundary

            days_to_next.append(delta_days)
            is_holiday_eve.append(1 if (0 <= delta_days <= 3) else 0)

            # Black Friday surge window (approx Nov 20 to Nov 30 in Brazil)
            is_bf = 1 if (ord_date.month == 11 and 20 <= ord_date.day <= 30) else 0
            is_bf_week.append(is_bf)

        return pd.DataFrame({
            "days_to_next_holiday": days_to_next,
            "is_holiday_eve": is_holiday_eve,
            "is_black_friday_week": is_bf_week
        }, index=approved_timestamps.index)

    @staticmethod
    def _get_fallback_holidays(year: int) -> List[Dict[str, Any]]:
        """Deterministic federal statutory Brazilian holidays for fallback."""
        fixed = [
            ("01-01", "Confraternização Universal"),
            ("04-21", "Tiradentes"),
            ("05-01", "Dia do Trabalhador"),
            ("09-07", "Independência do Brasil"),
            ("10-12", "Nossa Senhora Aparecida"),
            ("11-02", "Finados"),
            ("11-15", "Proclamação da República"),
            ("12-25", "Natal")
        ]
        return [{"date": f"{year}-{m_d}", "name": name, "type": "national", "year": year} for m_d, name in fixed]


class OpenMeteoWeatherBenchmarkClient:
    """
    Client for Open-Meteo Historical Weather Archive.
    Used exclusively for empirical benchmark comparisons and proving rejection rationale.
    """

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self.cache_dir = cache_dir or DEFAULT_CONFIG.external_data_dir
        self.cache_file = self.cache_dir / "weather_sample.json"

    def fetch_state_capital_weather_sample(
        self,
        latitude: float = -23.5505,
        longitude: float = -46.6333,
        start_date: str = "2017-01-01",
        end_date: str = "2017-01-31"
    ) -> Dict[str, Any]:
        """Fetch historical daily precipitation sample for São Paulo capital."""
        if self.cache_file.exists():
            with open(self.cache_file, "r", encoding="utf-8") as f:
                return json.load(f)

        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={latitude}&longitude={longitude}&"
            f"start_date={start_date}&end_date={end_date}&"
            f"daily=precipitation_sum,temperature_2m_max&timezone=America%2FSao_Paulo"
        )
        headers = {"User-Agent": "OlistEarlyWarningBenchmark/1.0"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return data

    fetch_weather_sample = fetch_state_capital_weather_sample

