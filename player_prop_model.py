"""
Player Prop Betting Model

This module provides a statistical model for analyzing and valuing player prop bets
in sports betting. It includes functionality for calculating expected values,
probabilities, and identifying positive expected value (EV) opportunities.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
from enum import Enum
import statistics


class PropType(Enum):
    """Types of player prop bets"""
    POINTS = "points"
    REBOUNDS = "rebounds"
    ASSISTS = "assists"
    THREE_POINTERS = "three_pointers"
    PASSING_YARDS = "passing_yards"
    RUSHING_YARDS = "rushing_yards"
    RECEIVING_YARDS = "receiving_yards"
    TOUCHDOWNS = "touchdowns"
    HITS = "hits"
    HOME_RUNS = "home_runs"
    STRIKEOUTS = "strikeouts"


class BetDirection(Enum):
    """Direction of the bet"""
    OVER = "over"
    UNDER = "under"


@dataclass
class PlayerProp:
    """
    Represents a player prop bet
    
    Attributes:
        player_name: Name of the player
        prop_type: Type of prop (points, assists, etc.)
        line: The betting line (e.g., 25.5 points)
        over_odds: Odds for over bet (American format, e.g., -110)
        under_odds: Odds for under bet (American format, e.g., -110)
    """
    player_name: str
    prop_type: PropType
    line: float
    over_odds: int
    under_odds: int
    
    def __str__(self):
        return f"{self.player_name} {self.prop_type.value} {self.line}"


@dataclass
class PlayerStats:
    """
    Historical statistics for a player
    
    Attributes:
        player_name: Name of the player
        prop_type: Type of statistic
        recent_games: List of stat values from recent games
        season_average: Season average for this stat
    """
    player_name: str
    prop_type: PropType
    recent_games: List[float]
    season_average: Optional[float] = None
    
    def __post_init__(self):
        if self.season_average is None and self.recent_games:
            self.season_average = statistics.mean(self.recent_games)


@dataclass
class PropAnalysis:
    """
    Analysis results for a player prop bet
    
    Attributes:
        prop: The player prop being analyzed
        estimated_value: Estimated value for the stat
        probability_over: Estimated probability of going over
        probability_under: Estimated probability of going under
        expected_value_over: Expected value of betting over
        expected_value_under: Expected value of betting under
        recommended_bet: Recommended betting direction (if any)
    """
    prop: PlayerProp
    estimated_value: float
    probability_over: float
    probability_under: float
    expected_value_over: float
    expected_value_under: float
    recommended_bet: Optional[BetDirection] = None
    
    def __str__(self):
        rec = f" [RECOMMENDATION: {self.recommended_bet.value.upper()}]" if self.recommended_bet else ""
        return (f"Analysis for {self.prop}\n"
                f"  Estimated Value: {self.estimated_value:.2f}\n"
                f"  P(Over): {self.probability_over:.1%}, EV: {self.expected_value_over:+.2%}\n"
                f"  P(Under): {self.probability_under:.1%}, EV: {self.expected_value_under:+.2%}"
                f"{rec}")


class PlayerPropModel:
    """
    Statistical model for analyzing player prop bets
    
    This model calculates probabilities and expected values based on
    historical player performance and betting lines.
    """
    
    def __init__(self, min_ev_threshold: float = 0.05):
        """
        Initialize the model
        
        Args:
            min_ev_threshold: Minimum expected value to recommend a bet (default 5%)
        """
        self.min_ev_threshold = min_ev_threshold
    
    @staticmethod
    def american_odds_to_decimal(american_odds: int) -> float:
        """
        Convert American odds to decimal odds
        
        Args:
            american_odds: American format odds (e.g., -110, +150)
            
        Returns:
            Decimal odds
        """
        if american_odds > 0:
            return (american_odds / 100) + 1
        else:
            return (100 / abs(american_odds)) + 1
    
    @staticmethod
    def american_odds_to_implied_probability(american_odds: int) -> float:
        """
        Convert American odds to implied probability
        
        Args:
            american_odds: American format odds (e.g., -110, +150)
            
        Returns:
            Implied probability as a decimal (0-1)
        """
        if american_odds > 0:
            return 100 / (american_odds + 100)
        else:
            return abs(american_odds) / (abs(american_odds) + 100)
    
    def estimate_probability_over(self, stats: PlayerStats, line: float) -> float:
        """
        Estimate the probability that a player will go over the line
        
        Uses recent game statistics to estimate probability
        
        Args:
            stats: Player's historical statistics
            line: The betting line
            
        Returns:
            Estimated probability (0-1)
        """
        if not stats.recent_games:
            return 0.5  # Default to 50% if no data
        
        # Calculate how many times player went over the line
        times_over = sum(1 for game in stats.recent_games if game > line)
        probability = times_over / len(stats.recent_games)
        
        # Apply smoothing to avoid extreme probabilities
        smoothing_factor = 0.1
        return (probability * (1 - smoothing_factor)) + (0.5 * smoothing_factor)
    
    def calculate_expected_value(self, probability: float, odds: int) -> float:
        """
        Calculate expected value of a bet
        
        Args:
            probability: True probability of winning (0-1)
            odds: American format odds
            
        Returns:
            Expected value as a percentage (e.g., 0.05 = 5% EV)
        """
        decimal_odds = self.american_odds_to_decimal(odds)
        # EV = (probability * profit) - (1 - probability) * stake
        # Assuming stake = 1, profit = decimal_odds - 1
        ev = (probability * (decimal_odds - 1)) - (1 - probability)
        return ev
    
    def analyze_prop(self, prop: PlayerProp, stats: PlayerStats) -> PropAnalysis:
        """
        Analyze a player prop bet and calculate expected value
        
        Args:
            prop: The player prop to analyze
            stats: Historical statistics for the player
            
        Returns:
            PropAnalysis object with complete analysis
        """
        # Estimate probabilities
        prob_over = self.estimate_probability_over(stats, prop.line)
        prob_under = 1 - prob_over
        
        # Calculate expected values
        ev_over = self.calculate_expected_value(prob_over, prop.over_odds)
        ev_under = self.calculate_expected_value(prob_under, prop.under_odds)
        
        # Determine recommendation
        recommendation = None
        if ev_over > self.min_ev_threshold:
            recommendation = BetDirection.OVER
        elif ev_under > self.min_ev_threshold:
            recommendation = BetDirection.UNDER
        
        # Use season average as estimated value, or recent average if not available
        estimated_value = stats.season_average if stats.season_average else (
            statistics.mean(stats.recent_games) if stats.recent_games else prop.line
        )
        
        return PropAnalysis(
            prop=prop,
            estimated_value=estimated_value,
            probability_over=prob_over,
            probability_under=prob_under,
            expected_value_over=ev_over,
            expected_value_under=ev_under,
            recommended_bet=recommendation
        )
    
    def find_value_bets(self, props: List[PlayerProp], 
                       stats_dict: Dict[str, PlayerStats]) -> List[PropAnalysis]:
        """
        Find all value bets from a list of props
        
        Args:
            props: List of player props to analyze
            stats_dict: Dictionary mapping player names to their stats
            
        Returns:
            List of PropAnalysis objects for bets with positive EV
        """
        value_bets = []
        
        for prop in props:
            if prop.player_name in stats_dict:
                stats = stats_dict[prop.player_name]
                # Only analyze if prop type matches stats type
                if stats.prop_type == prop.prop_type:
                    analysis = self.analyze_prop(prop, stats)
                    if analysis.recommended_bet:
                        value_bets.append(analysis)
        
        # Sort by expected value (highest first)
        value_bets.sort(
            key=lambda x: max(x.expected_value_over, x.expected_value_under),
            reverse=True
        )
        
        return value_bets
