"""Odds conversion and betting math utilities for NBA 3PM Props Engine.

All functions accept and return plain floats.  American odds follow the
standard convention: positive values for underdogs (e.g. +150), negative
values for favourites (e.g. -130).
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# American <-> Decimal conversions
# ---------------------------------------------------------------------------

def american_to_decimal(american: float) -> float:
    """Convert American odds to decimal odds.

    Examples
    --------
    >>> american_to_decimal(-110)
    1.9090909090909092
    >>> american_to_decimal(150)
    2.5
    """
    if american >= 100:
        return 1.0 + american / 100.0
    elif american <= -100:
        return 1.0 + 100.0 / abs(american)
    else:
        raise ValueError(
            f"Invalid American odds: {american}.  "
            "Must be <= -100 or >= +100."
        )


def decimal_to_american(decimal_odds: float) -> float:
    """Convert decimal odds to American odds.

    Examples
    --------
    >>> decimal_to_american(2.5)
    150.0
    >>> decimal_to_american(1.5)
    -200.0
    """
    if decimal_odds < 1.0:
        raise ValueError(
            f"Invalid decimal odds: {decimal_odds}.  Must be >= 1.0."
        )
    if decimal_odds >= 2.0:
        return (decimal_odds - 1.0) * 100.0
    else:
        return -100.0 / (decimal_odds - 1.0)


# ---------------------------------------------------------------------------
# American <-> Implied probability
# ---------------------------------------------------------------------------

def american_to_implied(american: float) -> float:
    """Convert American odds to raw implied probability (includes vig).

    Examples
    --------
    >>> round(american_to_implied(-110), 6)
    0.52381
    >>> round(american_to_implied(150), 6)
    0.4
    """
    if american <= -100:
        return abs(american) / (abs(american) + 100.0)
    elif american >= 100:
        return 100.0 / (american + 100.0)
    else:
        raise ValueError(
            f"Invalid American odds: {american}.  "
            "Must be <= -100 or >= +100."
        )


def implied_to_american(prob: float) -> float:
    """Convert an implied probability to American odds.

    Parameters
    ----------
    prob : float
        Probability in (0, 1).

    Examples
    --------
    >>> implied_to_american(0.6)
    -150.0
    >>> implied_to_american(0.4)
    150.0
    """
    if prob <= 0.0 or prob >= 1.0:
        raise ValueError(
            f"Probability must be in (0, 1), got {prob}."
        )
    if prob >= 0.5:
        return -(prob / (1.0 - prob)) * 100.0
    else:
        return ((1.0 - prob) / prob) * 100.0


# ---------------------------------------------------------------------------
# Vig removal (devigging)
# ---------------------------------------------------------------------------

def remove_vig_power(
    over_price: float,
    under_price: float,
) -> tuple[float, float]:
    """Remove vig using the *power method* (Shin-style).

    The power method finds an exponent *k* such that
    ``p_over^k + p_under^k = 1``, where ``p_over`` and ``p_under`` are
    the raw implied probabilities.  This method is considered more
    accurate for two-way markets than simple multiplicative scaling.

    Parameters
    ----------
    over_price, under_price : float
        American odds for each side.

    Returns
    -------
    tuple[float, float]
        ``(fair_prob_over, fair_prob_under)`` summing to 1.0.
    """
    p_over = american_to_implied(over_price)
    p_under = american_to_implied(under_price)

    # Binary search for exponent k where p_over^k + p_under^k == 1
    lo, hi = 0.0, 10.0
    for _ in range(200):  # plenty of iterations for convergence
        mid = (lo + hi) / 2.0
        total = p_over ** mid + p_under ** mid
        if total > 1.0:
            lo = mid
        else:
            hi = mid

    k = (lo + hi) / 2.0
    fair_over = p_over ** k
    fair_under = p_under ** k

    # Normalise to guarantee they sum to exactly 1.0
    s = fair_over + fair_under
    return fair_over / s, fair_under / s


def remove_vig_multiplicative(
    over_price: float,
    under_price: float,
) -> tuple[float, float]:
    """Remove vig using simple multiplicative (proportional) scaling.

    Each raw implied probability is divided by the sum of both implied
    probabilities so they sum to 1.0.

    Parameters
    ----------
    over_price, under_price : float
        American odds for each side.

    Returns
    -------
    tuple[float, float]
        ``(fair_prob_over, fair_prob_under)`` summing to 1.0.
    """
    p_over = american_to_implied(over_price)
    p_under = american_to_implied(under_price)
    total = p_over + p_under
    return p_over / total, p_under / total


# ---------------------------------------------------------------------------
# Hold / overround
# ---------------------------------------------------------------------------

def compute_hold(over_price: float, under_price: float) -> float:
    """Compute the sportsbook hold (overround) for a two-way market.

    Returns the hold as a fraction (e.g. 0.0476 for ~4.76% hold).

    Examples
    --------
    >>> round(compute_hold(-110, -110), 4)
    0.0476
    """
    p_over = american_to_implied(over_price)
    p_under = american_to_implied(under_price)
    return p_over + p_under - 1.0


# ---------------------------------------------------------------------------
# Expected value and Kelly sizing
# ---------------------------------------------------------------------------

def compute_ev(model_prob: float, decimal_odds: float) -> float:
    """Compute the expected value of a bet.

    Parameters
    ----------
    model_prob : float
        Model's estimated probability of the outcome.
    decimal_odds : float
        Decimal odds offered by the book.

    Returns
    -------
    float
        EV as a fraction of stake.  Positive = +EV.
    """
    return model_prob * decimal_odds - 1.0


def kelly_criterion(model_prob: float, decimal_odds: float) -> float:
    """Compute the full Kelly stake fraction.

    Parameters
    ----------
    model_prob : float
        Model's estimated probability of winning.
    decimal_odds : float
        Decimal odds offered.

    Returns
    -------
    float
        Optimal fraction of bankroll to stake.  Clamped to >= 0
        (never recommends a negative stake).
    """
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    f = (model_prob * b - (1.0 - model_prob)) / b
    return max(f, 0.0)


def half_kelly(model_prob: float, decimal_odds: float) -> float:
    """Compute half-Kelly stake fraction.

    A more conservative sizing approach that uses half of the full
    Kelly fraction.

    Parameters
    ----------
    model_prob : float
        Model's estimated probability of winning.
    decimal_odds : float
        Decimal odds offered.

    Returns
    -------
    float
        Half-Kelly fraction, clamped to >= 0.
    """
    return kelly_criterion(model_prob, decimal_odds) / 2.0
