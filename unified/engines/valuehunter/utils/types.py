"""Core domain types for NBA 3PM Props Engine."""

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
    SECOND_SPECTRUM = "second_spectrum"  # 25 samples/sec
    HAWKEYE = "hawkeye"  # 29 body points, 60 samples/sec


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
class InjurySnapshot:
    injury_report_id: Optional[int]
    nba_game_id: str
    nba_player_id: str
    team_abbr: str
    report_timestamp_utc: datetime
    report_source: str
    status: InjuryStatus
    reason_text: Optional[str] = None
    report_url: Optional[str] = None
    report_hash: Optional[str] = None
    minutes_limit_flag: bool = False


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


@dataclass
class MinutesPrediction:
    p10: float
    p50: float
    p90: float


@dataclass
class ThreePAPrediction:
    mean: float
    dispersion: float


@dataclass
class MakeRatePrediction:
    mean: float
    uncertainty: float


@dataclass
class PricingResult:
    p_over: float
    p_under: float
    fair_odds_over: float
    fair_odds_under: float
    mean_3pm: float
    median_3pm: float
    p10_3pm: float
    p90_3pm: float
    simulations: int


@dataclass
class BetDecision:
    decision_id: Optional[int]
    feature_snapshot_id: int
    model_run_id: int
    nba_game_id: str
    nba_player_id: str
    sportsbook: str
    line: float
    odds_over: float
    odds_under: float
    model_p_over: float
    model_p_under: float
    fair_odds_over: float
    fair_odds_under: float
    edge_over: float
    edge_under: float
    recommended_side: BetSide
    stake_pct: float
    decision_timestamp_utc: datetime
    tracking_available: bool = False
    close_over_prob_novig: Optional[float] = None
    close_under_prob_novig: Optional[float] = None
    clv_prob_pts: Optional[float] = None
    actual_3pm: Optional[int] = None
    bet_result: Optional[BetResult] = None
    pnl_units: Optional[float] = None


@dataclass(frozen=True)
class FeatureLineage:
    """Records source family and era for a feature."""
    feature_name: str
    source_family: str  # e.g., "tracking", "boxscore", "opponent_shooting"
    tracking_era: Optional[TrackingEra] = None
    provider: Optional[TrackingProvider] = None
    is_imputed: bool = False
    coverage_flag: bool = True  # True if data was available
