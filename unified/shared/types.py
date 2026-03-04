"""Shared domain types used across all engines.

From: ValueHunter/src/nba_props/utils/types.py
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional


class SeasonType(str, enum.Enum):
    REGULAR = "Regular Season"
    PLAYOFFS = "Playoffs"


class PlayerArchetype(str, enum.Enum):
    MOVEMENT_WING_SHOOTER = "movement_wing_shooter"
    PULL_UP_GUARD = "pull_up_guard"
    STRETCH_BIG = "stretch_big"
    STATIONARY_SPACER = "stationary_spacer"
    BENCH_MICROWAVE = "bench_microwave"


class InjuryStatus(str, enum.Enum):
    AVAILABLE = "Available"
    QUESTIONABLE = "Questionable"
    DOUBTFUL = "Doubtful"
    OUT = "Out"
    PROBABLE = "Probable"
    GTD = "Game Time Decision"


class BetSide(str, enum.Enum):
    OVER = "over"
    UNDER = "under"
    NO_BET = "no_bet"


class BetResult(str, enum.Enum):
    WIN = "win"
    LOSS = "loss"
    PUSH = "push"
    PENDING = "pending"


class TrackingProvider(str, enum.Enum):
    SECOND_SPECTRUM = "second_spectrum"
    HAWKEYE_AWS = "hawkeye_aws"
    UNKNOWN = "unknown"


class TrackingEra(str, enum.Enum):
    SECOND_SPECTRUM = "second_spectrum"
    HAWKEYE = "hawkeye"


class SourceMode(str, enum.Enum):
    """Data source mode for adapter selection."""
    PREMIUM = "premium"
    FALLBACK = "fallback"
    MOCK = "mock"


@dataclass(frozen=True)
class GameInfo:
    nba_game_id: str
    season: str
    season_type: SeasonType
    game_date: date
    tipoff_time_utc: Optional[datetime]
    home_team_abbr: str
    away_team_abbr: str
    sr_game_id: Optional[str] = None
    closing_spread_home: Optional[float] = None
    closing_total: Optional[float] = None


@dataclass(frozen=True)
class PlayerGameRow:
    nba_game_id: str
    nba_player_id: str
    team_abbr: str
    opponent_abbr: str
    is_home: bool
    started: bool
    minutes_played: float
    three_pa: int
    three_pm: int
    fg3_pct: Optional[float]
    usage_rate: Optional[float]
    assist_rate: Optional[float]
    turnovers: int = 0
    personal_fouls: int = 0
    rest_days: Optional[int] = None
    is_back_to_back: bool = False
    is_3in4: bool = False


@dataclass(frozen=True)
class TrackingRow:
    nba_game_id: str
    nba_player_id: str
    tracking_available: bool
    tracking_provider: Optional[TrackingProvider] = None
    touches: Optional[float] = None
    passes_made: Optional[int] = None
    passes_received: Optional[int] = None
    time_of_possession_sec: Optional[float] = None
    avg_seconds_per_touch: Optional[float] = None
    avg_dribbles_per_touch: Optional[float] = None
    catch_shoot_fga: Optional[int] = None
    catch_shoot_fgm: Optional[int] = None
    catch_shoot_3pa: Optional[int] = None
    catch_shoot_3pm: Optional[int] = None
    pull_up_fga: Optional[int] = None
    pull_up_fgm: Optional[int] = None
    pull_up_3pa: Optional[int] = None
    pull_up_3pm: Optional[int] = None
    potential_assists: Optional[int] = None
    secondary_assists: Optional[int] = None


@dataclass(frozen=True)
class OddsProp:
    odds_prop_id: Optional[int]
    snapshot_timestamp_utc: datetime
    sportsbook: str
    market: str
    nba_game_id: str
    nba_player_id: str
    line: float
    over_price: float
    under_price: float
    over_implied_prob_raw: Optional[float] = None
    under_implied_prob_raw: Optional[float] = None
    over_implied_prob_novig: Optional[float] = None
    under_implied_prob_novig: Optional[float] = None
    hold_pct: Optional[float] = None
    is_closing_snapshot: bool = False
    source: Optional[str] = None
