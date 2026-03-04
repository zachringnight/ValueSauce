from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set

Market = str
Status = str

VALID_MARKETS: Set[str] = {"PTS", "REB", "AST", "FG3M"}
VALID_STATUSES: Set[str] = {"OUT", "DOUBTFUL", "QUESTIONABLE", "PROBABLE", "AVAILABLE", "UNKNOWN"}

@dataclass(frozen=True)
class MinutesOverride:
    kind: str  # cap | target
    value: float

@dataclass
class GameOverrides:
    team_out: Set[str] = field(default_factory=set)
    team_in: Set[str] = field(default_factory=set)
    opp_out: Set[str] = field(default_factory=set)
    opp_in: Set[str] = field(default_factory=set)
    minutes_overrides: Dict[str, MinutesOverride] = field(default_factory=dict)
    notes: str = ""

@dataclass
class Scenario:
    name: str
    prob: float
    team_out: Set[str] = field(default_factory=set)
    team_in: Set[str] = field(default_factory=set)
    opp_out: Set[str] = field(default_factory=set)
    opp_in: Set[str] = field(default_factory=set)

@dataclass
class GameState:
    game_key: str
    asof_utc: datetime
    game_date_local: Optional[str]
    player_team_abbr: str
    opponent_abbr: str
    is_home: bool
    overrides: GameOverrides
    spread: Optional[float] = None
    vegas_team_total: Optional[float] = None
    vegas_game_total: Optional[float] = None

    player_team_id: Optional[int] = None
    opponent_team_id: Optional[int] = None
    player_team_name: Optional[str] = None
    opponent_team_name: Optional[str] = None

    # Mapped statuses
    resolved_status_by_name: Dict[str, Status] = field(default_factory=dict)
    resolved_status_by_player_id: Dict[int, Status] = field(default_factory=dict)
    unmapped_injury_names: List[str] = field(default_factory=list)

    # Rosters / identities
    team_roster_by_name: Dict[str, int] = field(default_factory=dict)
    opp_roster_by_name: Dict[str, int] = field(default_factory=dict)

    scenarios: List[Scenario] = field(default_factory=list)

@dataclass(frozen=True)
class PropRequest:
    game_key: str
    player_name: str
    market: Market
    line: float
    odds_american: Optional[int]
    odds_over_american: Optional[int] = None
    odds_under_american: Optional[int] = None
    bookmaker: Optional[str] = None
    event_id: Optional[str] = None
    event_start_utc: Optional[str] = None

@dataclass(frozen=True)
class DistributionSpec:
    name: str  # normal | neg_binom
    params: Dict[str, float]

@dataclass(frozen=True)
class ProjectionResult:
    game_key: str
    asof_utc: datetime
    player_name: str
    market: Market
    line: float
    odds_american: Optional[int]

    mu: float
    sigma: float
    dist: DistributionSpec

    raw_p_over: float
    p_over: float
    p_under: float

    fair_over_odds: Optional[int]
    edge_cents_over: Optional[float]

    flags: List[str]
    drivers: List[str]
    scenario_details: List[Dict[str, float]]

    odds_over_american: Optional[int] = None
    odds_under_american: Optional[int] = None
    bookmaker: Optional[str] = None
    event_id: Optional[str] = None
    event_start_utc: Optional[str] = None
    fair_under_odds: Optional[int] = None
    market_prob_over: Optional[float] = None
    market_prob_under: Optional[float] = None
    edge_cents_under: Optional[float] = None
    recommended_side: Optional[str] = None
    recommended_odds_american: Optional[int] = None
    model_prob_side: Optional[float] = None
    market_prob_side: Optional[float] = None
    edge_cents_side: Optional[float] = None
    eligible_for_recommendation: bool = False
