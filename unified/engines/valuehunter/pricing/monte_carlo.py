"""Monte Carlo simulation for 3PM distributional pricing."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Results from Monte Carlo simulation for a single player-game."""
    p_over: float
    p_under: float
    fair_odds_over_american: float
    fair_odds_under_american: float
    fair_odds_over_decimal: float
    fair_odds_under_decimal: float
    mean_3pm: float
    median_3pm: float
    p10_3pm: float
    p90_3pm: float
    line: float
    simulations: int
    alt_line_probs: dict  # {line: {"p_over": float, "p_under": float}}


class MonteCarloSimulator:
    """
    Monte Carlo simulator for 3PM distribution.

    Pipeline per player x game x snapshot:
    1. Draw minutes from minutes distribution
    2. Draw 3PA conditional on minutes from count model
    3. Draw make probability conditional on context
    4. Draw 3PM from Binomial(3PA, make_prob)
    5. Repeat n_simulations times
    """

    def __init__(self, n_simulations: int = 25000, seed: Optional[int] = None):
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(seed)

    def simulate(
        self,
        minutes_p10: float,
        minutes_p50: float,
        minutes_p90: float,
        three_pa_mean: float,
        three_pa_dispersion: float,
        make_prob_mean: float,
        make_prob_uncertainty: float,
        line: float,
        alt_lines: Optional[list[float]] = None,
    ) -> SimulationResult:
        """
        Run full Monte Carlo simulation for a single player-game.

        Args:
            minutes_p10: 10th percentile minutes prediction
            minutes_p50: Median minutes prediction
            minutes_p90: 90th percentile minutes prediction
            three_pa_mean: Mean 3PA prediction (per-game, not per-minute)
            three_pa_dispersion: Dispersion parameter for negative binomial
            make_prob_mean: Mean make probability
            make_prob_uncertainty: Uncertainty in make probability (std of Beta)
            line: The prop line (e.g., 2.5)
            alt_lines: Optional list of alternative lines to price

        Returns:
            SimulationResult with all pricing outputs
        """
        if alt_lines is None:
            alt_lines = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]

        # Step 1: Draw minutes from a log-normal fitted to quantiles
        minutes_draws = self._draw_minutes(minutes_p10, minutes_p50, minutes_p90)

        # Step 2: Draw 3PA conditional on minutes
        # Scale mean 3PA by minutes ratio (drawn_minutes / expected_minutes)
        minutes_ratio = np.clip(minutes_draws / max(minutes_p50, 1.0), 0.0, 2.0)
        conditional_3pa_mean = three_pa_mean * minutes_ratio

        three_pa_draws = self._draw_3pa(conditional_3pa_mean, three_pa_dispersion)

        # Step 3: Draw make probability from Beta distribution
        make_prob_draws = self._draw_make_prob(make_prob_mean, make_prob_uncertainty)

        # Step 4: Draw 3PM from Binomial(3PA, make_prob)
        three_pm_draws = self.rng.binomial(
            n=three_pa_draws.astype(int),
            p=np.clip(make_prob_draws, 0.001, 0.999),
        )

        # Compute results
        p_over = float(np.mean(three_pm_draws > line))
        p_under = float(np.mean(three_pm_draws <= line))

        # Ensure probabilities sum to 1 (handle floating point)
        total = p_over + p_under
        if total > 0:
            p_over /= total
            p_under /= total

        # Fair odds
        fair_over_decimal = 1.0 / max(p_over, 0.001)
        fair_under_decimal = 1.0 / max(p_under, 0.001)
        fair_over_american = self._decimal_to_american(fair_over_decimal)
        fair_under_american = self._decimal_to_american(fair_under_decimal)

        # Alt line probabilities
        alt_line_probs = {}
        for alt_line in alt_lines:
            alt_p_over = float(np.mean(three_pm_draws > alt_line))
            alt_p_under = 1.0 - alt_p_over
            alt_line_probs[alt_line] = {
                "p_over": alt_p_over,
                "p_under": alt_p_under,
            }

        return SimulationResult(
            p_over=p_over,
            p_under=p_under,
            fair_odds_over_american=fair_over_american,
            fair_odds_under_american=fair_under_american,
            fair_odds_over_decimal=fair_over_decimal,
            fair_odds_under_decimal=fair_under_decimal,
            mean_3pm=float(np.mean(three_pm_draws)),
            median_3pm=float(np.median(three_pm_draws)),
            p10_3pm=float(np.percentile(three_pm_draws, 10)),
            p90_3pm=float(np.percentile(three_pm_draws, 90)),
            line=line,
            simulations=self.n_simulations,
            alt_line_probs=alt_line_probs,
        )

    def _draw_minutes(
        self, p10: float, p50: float, p90: float
    ) -> np.ndarray:
        """Draw minutes from log-normal distribution fitted to quantiles."""
        # Fit log-normal to p50 (median) and spread from p10/p90
        p50 = max(p50, 1.0)
        p10 = max(p10, 0.0)
        p90 = max(p90, p50 + 1.0)

        # Log-normal: median = exp(mu), so mu = log(median)
        mu = np.log(p50)

        # Use IQR-like approach: p90/p50 ratio to estimate sigma
        if p50 > 0:
            ratio = p90 / p50
            sigma = np.log(max(ratio, 1.01)) / 1.2816  # z-score for 90th percentile
        else:
            sigma = 0.3

        sigma = np.clip(sigma, 0.05, 1.0)

        draws = self.rng.lognormal(mean=mu, sigma=sigma, size=self.n_simulations)
        # Cap at 48 minutes (regulation) and floor at 0
        draws = np.clip(draws, 0.0, 48.0)
        return draws

    def _draw_3pa(
        self, conditional_mean: np.ndarray, dispersion: float
    ) -> np.ndarray:
        """
        Draw 3PA from Negative Binomial distribution.

        Uses the mean-dispersion parameterization:
        - mean = mu
        - variance = mu + mu^2 / dispersion

        Higher dispersion -> closer to Poisson (less overdispersion).
        """
        dispersion = max(dispersion, 0.1)
        mu = np.clip(conditional_mean, 0.01, 30.0)

        # Convert to numpy's parameterization: n, p
        # n = dispersion, p = dispersion / (dispersion + mu)
        n = dispersion
        p = dispersion / (dispersion + mu)
        p = np.clip(p, 0.001, 0.999)

        draws = self.rng.negative_binomial(n=np.full(self.n_simulations, n), p=p)
        return draws.astype(float)

    def _draw_make_prob(
        self, mean: float, uncertainty: float
    ) -> np.ndarray:
        """
        Draw make probability from Beta distribution.

        Parameterized by mean and uncertainty (std dev).
        """
        mean = np.clip(mean, 0.01, 0.99)
        uncertainty = np.clip(uncertainty, 0.005, 0.2)

        # Convert mean and variance to alpha, beta
        # For Beta: mean = alpha / (alpha + beta)
        # variance = alpha*beta / ((alpha+beta)^2 * (alpha+beta+1))
        variance = uncertainty ** 2

        # Solve for concentration = alpha + beta
        # variance = mean * (1-mean) / (concentration + 1)
        # concentration = mean * (1-mean) / variance - 1
        concentration = mean * (1 - mean) / max(variance, 1e-6) - 1
        concentration = np.clip(concentration, 2.0, 1000.0)

        alpha = mean * concentration
        beta = (1 - mean) * concentration

        draws = self.rng.beta(a=alpha, b=beta, size=self.n_simulations)
        return np.clip(draws, 0.001, 0.999)

    @staticmethod
    def _decimal_to_american(decimal_odds: float) -> float:
        """Convert decimal odds to American odds."""
        if decimal_odds >= 2.0:
            return round((decimal_odds - 1) * 100, 1)
        elif decimal_odds > 1.0:
            return round(-100 / (decimal_odds - 1), 1)
        else:
            return -10000.0
