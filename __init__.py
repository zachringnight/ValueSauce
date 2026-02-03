"""
ValueSauce - Player Prop Betting Model

A Python package for analyzing player prop bets and finding value
in sports betting markets.
"""

from .player_prop_model import (
    PlayerPropModel,
    PlayerProp,
    PlayerStats,
    PropAnalysis,
    PropType,
    BetDirection
)

__version__ = "0.1.0"
__author__ = "ValueSauce"

__all__ = [
    "PlayerPropModel",
    "PlayerProp",
    "PlayerStats",
    "PropAnalysis",
    "PropType",
    "BetDirection",
]
