"""Centralized settings loaded from environment / .env file."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class SourceMode(str, Enum):
    PREMIUM = "premium"
    FALLBACK = "fallback"
    MOCK = "mock"


class Settings(BaseSettings):
    # Source mode
    source_mode: SourceMode = SourceMode.PREMIUM

    # Database
    database_url: str = "sqlite:///data/nba_props.db"
    duckdb_path: str = "data/nba_props.duckdb"

    # Premium adapters
    premium_nba_api_key: str = ""
    premium_nba_base_url: str = ""
    premium_possession_api_key: str = ""
    premium_possession_base_url: str = ""
    premium_odds_api_key: str = ""
    premium_odds_base_url: str = ""

    # SportsGameOdds API (for real odds data)
    sportsgameodds_api_key: str = ""

    # The Odds API (alternative odds source)
    the_odds_api_key: str = ""

    # Official/public
    nba_stats_base_url: str = "https://stats.nba.com/stats"

    # Simulation
    monte_carlo_draws: int = 10_000
    monte_carlo_seed: int = 42

    # Bet rules
    min_projected_minutes: float = 24.0
    min_edge_threshold: float = 0.035
    max_bets_per_player: int = 1
    max_correlated_bets_per_game: int = 2
    paper_trade_mode: bool = True

    # Logging
    log_level: str = "INFO"
    log_dir: str = "logs"

    # Paths
    data_dir: str = "data"
    sample_payloads_dir: str = "sample_payloads"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def raw_dir(self) -> Path:
        return Path(self.data_dir) / "raw"

    @property
    def interim_dir(self) -> Path:
        return Path(self.data_dir) / "interim"

    @property
    def processed_dir(self) -> Path:
        return Path(self.data_dir) / "processed"

    @property
    def backtests_dir(self) -> Path:
        return Path(self.data_dir) / "backtests"

    @property
    def reports_dir(self) -> Path:
        return Path(self.data_dir) / "reports"

    def config_hash(self) -> str:
        """Deterministic hash of model-relevant config values."""
        relevant = {
            "source_mode": self.source_mode.value,
            "monte_carlo_draws": self.monte_carlo_draws,
            "monte_carlo_seed": self.monte_carlo_seed,
            "min_projected_minutes": self.min_projected_minutes,
            "min_edge_threshold": self.min_edge_threshold,
        }
        return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:16]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
