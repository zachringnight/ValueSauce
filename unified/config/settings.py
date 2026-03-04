"""Unified settings covering all API keys and engine thresholds.

Merges:
- ValueHunter/src/nba_props/config/settings.py
- nil/nba_props/config/settings.py
- NBA_Props_AI env config
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Settings:
    # ---------------------------------------------------------------
    # Database
    # ---------------------------------------------------------------
    database_url: str = "sqlite:///data/cache.sqlite"

    # ---------------------------------------------------------------
    # NBA Stats
    # ---------------------------------------------------------------
    nba_stats_base_url: str = "https://stats.nba.com/stats"
    nba_stats_rate_limit_sec: float = 0.6

    # ---------------------------------------------------------------
    # SportsGameOdds
    # ---------------------------------------------------------------
    sportsgameodds_api_key: str = ""

    # ---------------------------------------------------------------
    # The Odds API
    # ---------------------------------------------------------------
    the_odds_api_key: str = ""
    odds_api_base_url: str = "https://api.the-odds-api.com/v4"
    odds_api_history_start: str = "2023-05-03"

    # ---------------------------------------------------------------
    # TheRundown (NEW)
    # ---------------------------------------------------------------
    therundown_api_key: str = ""

    # ---------------------------------------------------------------
    # Sportradar
    # ---------------------------------------------------------------
    sportradar_api_key: str = ""
    sportradar_base_url: str = "https://api.sportradar.us/nba/trial/v8/en"

    # ---------------------------------------------------------------
    # Injury reports
    # ---------------------------------------------------------------
    nba_injury_report_url: str = "https://official.nba.com/nba-injury-report-2025-26-season/"

    # ---------------------------------------------------------------
    # Source mode
    # ---------------------------------------------------------------
    source_mode: str = "fallback"  # premium | fallback | mock

    # ---------------------------------------------------------------
    # NBA_Props_AI thresholds
    # ---------------------------------------------------------------
    nba_props_ai_monte_carlo_draws: int = 25000
    nba_props_ai_min_projected_minutes: float = 18.0
    nba_props_ai_min_ev_pct: float = 0.03
    nba_props_ai_min_edge_prob_pts: float = 0.025
    nba_props_ai_max_hold_pct: float = 0.12

    # ---------------------------------------------------------------
    # ValueHunter thresholds
    # ---------------------------------------------------------------
    valuehunter_monte_carlo_draws: int = 25000
    valuehunter_min_projected_minutes: float = 18.0
    valuehunter_min_season_avg_minutes: float = 16.0
    valuehunter_min_ev_pct: float = 0.03
    valuehunter_min_edge_prob_pts: float = 0.025
    valuehunter_max_hold_pct: float = 0.12
    valuehunter_flat_stake_pct: float = 0.005
    valuehunter_max_kelly_stake_pct: float = 0.005
    valuehunter_max_correlated_positions: int = 3

    # ---------------------------------------------------------------
    # nil thresholds
    # ---------------------------------------------------------------
    nil_monte_carlo_draws: int = 10000
    nil_monte_carlo_seed: int = 42
    nil_min_projected_minutes: float = 24.0
    nil_min_edge_threshold: float = 0.035
    nil_max_bets_per_player: int = 1
    nil_max_correlated_bets_per_game: int = 2
    nil_paper_trade_mode: bool = True

    # ---------------------------------------------------------------
    # Tracking
    # ---------------------------------------------------------------
    tracking_providers: tuple = ("second_spectrum", "hawkeye_aws")

    # ---------------------------------------------------------------
    # API server
    # ---------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ---------------------------------------------------------------
    # Paths
    # ---------------------------------------------------------------
    data_dir: str = "data"
    cache_db_path: str = "data/cache.sqlite"

    @classmethod
    def from_env(cls) -> Settings:
        """Load settings from environment variables with sensible defaults."""
        return cls(
            database_url=os.getenv("DATABASE_URL", cls.database_url),
            sportsgameodds_api_key=os.getenv("SPORTSGAMEODDS_API_KEY", ""),
            the_odds_api_key=os.getenv("THE_ODDS_API_KEY", ""),
            therundown_api_key=os.getenv("THERUNDOWN_API_KEY", ""),
            sportradar_api_key=os.getenv("SPORTRADAR_API_KEY", ""),
            source_mode=os.getenv("SOURCE_MODE", "fallback"),
            api_host=os.getenv("API_HOST", cls.api_host),
            api_port=int(os.getenv("API_PORT", str(cls.api_port))),
            data_dir=os.getenv("DATA_DIR", cls.data_dir),
            cache_db_path=os.getenv("CACHE_DB_PATH", cls.cache_db_path),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton settings."""
    return Settings.from_env()
