"""Application settings loaded from environment variables."""

import os
from dataclasses import dataclass, field
from functools import lru_cache


@dataclass(frozen=True)
class Settings:
    # Database
    database_url: str = "postgresql://nba_props:nba_props@localhost:5432/nba_props"

    # NBA Stats
    nba_stats_base_url: str = "https://stats.nba.com/stats"
    nba_stats_rate_limit_sec: float = 0.6

    # Sportradar
    sportradar_api_key: str = ""
    sportradar_base_url: str = "https://api.sportradar.us/nba/production/v8"

    # The Odds API
    odds_api_key: str = ""
    odds_api_base_url: str = "https://api.the-odds-api.com/v4"
    odds_api_history_start: str = "2023-05-03"

    # Injury reports
    nba_injury_report_url: str = "https://official.nba.com/nba-injury-report-2025-26-season/"

    # Model settings
    monte_carlo_draws: int = 25000
    min_projected_minutes: float = 18.0
    min_trailing_10g_avg_minutes: float = 16.0
    min_season_avg_minutes: float = 16.0
    min_trailing_20g_3pa_per_36: float = 4.5
    min_prop_appearances: int = 8
    prop_appearance_window: int = 15

    # Decision thresholds
    min_ev_pct: float = 0.03
    min_edge_prob_pts: float = 0.025
    max_stale_odds_minutes: int = 30
    max_hold_pct: float = 0.12

    # Risk management
    flat_stake_pct: float = 0.005
    max_kelly_stake_pct: float = 0.005
    max_game_exposure_pct: float = 0.02
    max_correlated_positions: int = 3
    paper_trade_min_bets: int = 100

    # Tracking
    tracking_providers: tuple = ("second_spectrum", "hawkeye_aws")

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            database_url=os.getenv("DATABASE_URL", cls.database_url),
            sportradar_api_key=os.getenv("SPORTRADAR_API_KEY", ""),
            odds_api_key=os.getenv("ODDS_API_KEY", ""),
            api_host=os.getenv("API_HOST", cls.api_host),
            api_port=int(os.getenv("API_PORT", str(cls.api_port))),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
