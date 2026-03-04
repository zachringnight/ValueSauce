"""Unified odds math — superset of all three repos.

Sources:
- ValueHunter: american_to_decimal, decimal_to_american, american_to_implied,
  implied_to_american, remove_vig_power, remove_vig_multiplicative,
  compute_hold, compute_ev, kelly_criterion, half_kelly
- nil: remove_vig, compute_edge
- NBA_Props_AI: american_to_implied_prob, implied_prob_to_american, eb_shrink_rate
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# American <-> Decimal conversions (ValueHunter)
# ---------------------------------------------------------------------------

def american_to_decimal(american: float) -> float:
    """Convert American odds to decimal odds."""
    if american >= 100:
        return 1.0 + american / 100.0
    elif american <= -100:
        return 1.0 + 100.0 / abs(american)
    else:
        raise ValueError(f"Invalid American odds: {american}. Must be <= -100 or >= +100.")


def decimal_to_american(decimal_odds: float) -> float:
    """Convert decimal odds to American odds."""
    if decimal_odds < 1.0:
        raise ValueError(f"Invalid decimal odds: {decimal_odds}. Must be >= 1.0.")
    if decimal_odds >= 2.0:
        return (decimal_odds - 1.0) * 100.0
    else:
        return -100.0 / (decimal_odds - 1.0)


# ---------------------------------------------------------------------------
# American <-> Implied probability
# ---------------------------------------------------------------------------

def american_to_implied(american: float) -> float:
    """Convert American odds to raw implied probability (includes vig)."""
    if american <= -100:
        return abs(american) / (abs(american) + 100.0)
    elif american >= 100:
        return 100.0 / (american + 100.0)
    else:
        raise ValueError(f"Invalid American odds: {american}. Must be <= -100 or >= +100.")


# Alias used by NBA_Props_AI engine
american_to_implied_prob = american_to_implied


def implied_to_american(prob: float) -> float:
    """Convert implied probability to American odds."""
    if prob <= 0.0 or prob >= 1.0:
        raise ValueError(f"Probability must be in (0, 1), got {prob}.")
    if prob >= 0.5:
        return -(prob / (1.0 - prob)) * 100.0
    else:
        return ((1.0 - prob) / prob) * 100.0


# Alias for NBA_Props_AI (returns int)
def implied_prob_to_american(p: float) -> int:
    """Convert probability to American odds (int)."""
    if not (0.0 < p < 1.0):
        raise ValueError("Probability must be in (0,1)")
    dec = 1.0 / p
    if dec >= 2.0:
        return int(round((dec - 1.0) * 100.0))
    return int(round(-100.0 / (dec - 1.0)))


# ---------------------------------------------------------------------------
# Vig removal (ValueHunter + nil)
# ---------------------------------------------------------------------------

def remove_vig_power(over_price: float, under_price: float) -> tuple[float, float]:
    """Remove vig using the power method (Shin-style). Binary search for exponent k."""
    p_over = american_to_implied(over_price)
    p_under = american_to_implied(under_price)

    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        total = p_over ** mid + p_under ** mid
        if total > 1.0:
            lo = mid
        else:
            hi = mid

    k = (lo + hi) / 2.0
    fair_over = p_over ** k
    fair_under = p_under ** k
    s = fair_over + fair_under
    return fair_over / s, fair_under / s


def remove_vig_multiplicative(over_price: float, under_price: float) -> tuple[float, float]:
    """Remove vig using simple multiplicative scaling."""
    p_over = american_to_implied(over_price)
    p_under = american_to_implied(under_price)
    total = p_over + p_under
    return p_over / total, p_under / total


def remove_vig(over_prob: float, under_prob: float) -> tuple[float, float]:
    """Remove vig from raw probabilities (nil's version)."""
    total = over_prob + under_prob
    if total <= 0:
        raise ValueError("Probabilities must be positive")
    return over_prob / total, under_prob / total


# ---------------------------------------------------------------------------
# Hold / overround (ValueHunter)
# ---------------------------------------------------------------------------

def compute_hold(over_price: float, under_price: float) -> float:
    """Compute sportsbook hold (overround) for a two-way market."""
    p_over = american_to_implied(over_price)
    p_under = american_to_implied(under_price)
    return p_over + p_under - 1.0


# ---------------------------------------------------------------------------
# Edge (nil)
# ---------------------------------------------------------------------------

def compute_edge(fair_prob: float, market_prob: float) -> float:
    """Edge = fair_prob - market_prob (positive = value)."""
    return fair_prob - market_prob


# ---------------------------------------------------------------------------
# Expected value and Kelly sizing (ValueHunter)
# ---------------------------------------------------------------------------

def compute_ev(model_prob: float, decimal_odds: float) -> float:
    """Expected value: model_prob * decimal_odds - 1.0."""
    return model_prob * decimal_odds - 1.0


def kelly_criterion(model_prob: float, decimal_odds: float) -> float:
    """Full Kelly stake fraction, clamped >= 0."""
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    f = (model_prob * b - (1.0 - model_prob)) / b
    return max(f, 0.0)


def half_kelly(model_prob: float, decimal_odds: float) -> float:
    """Half-Kelly stake fraction."""
    return kelly_criterion(model_prob, decimal_odds) / 2.0


# ---------------------------------------------------------------------------
# Empirical Bayes shrinkage (NBA_Props_AI)
# ---------------------------------------------------------------------------

def eb_shrink_rate(obs_rate: float, obs_n: float, prior_rate: float, prior_n: float) -> float:
    """Empirical Bayes shrinkage estimator."""
    if obs_n < 0:
        obs_n = 0
    if prior_n <= 0:
        return obs_rate
    return (obs_rate * obs_n + prior_rate * prior_n) / (obs_n + prior_n)


# ---------------------------------------------------------------------------
# Utility (NBA_Props_AI)
# ---------------------------------------------------------------------------

def safe_clip(x: float, lo: float, hi: float) -> float:
    """Clamp value between lo and hi."""
    return max(lo, min(hi, x))
