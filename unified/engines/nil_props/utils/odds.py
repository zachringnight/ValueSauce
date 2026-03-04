"""Odds conversion utilities."""

from __future__ import annotations


def american_to_implied(american: float) -> float:
    """Convert American odds to implied probability."""
    if american >= 100:
        return 100.0 / (american + 100.0)
    elif american <= -100:
        return abs(american) / (abs(american) + 100.0)
    else:
        raise ValueError(f"Invalid American odds: {american}")


def implied_to_american(prob: float) -> float:
    """Convert implied probability to American odds."""
    if prob <= 0 or prob >= 1:
        raise ValueError(f"Probability must be in (0, 1), got {prob}")
    if prob >= 0.5:
        return -100.0 * prob / (1.0 - prob)
    else:
        return 100.0 * (1.0 - prob) / prob


def remove_vig(over_prob: float, under_prob: float) -> tuple[float, float]:
    """Remove vig to get fair probabilities."""
    total = over_prob + under_prob
    if total <= 0:
        raise ValueError("Probabilities must be positive")
    return over_prob / total, under_prob / total


def compute_edge(fair_prob: float, market_prob: float) -> float:
    """Edge = fair_prob - market_prob (positive = value)."""
    return fair_prob - market_prob
