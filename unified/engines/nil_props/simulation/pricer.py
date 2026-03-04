"""Monte Carlo pricing engine for player assist props."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from engines.nil_props.utils.odds import implied_to_american

logger = logging.getLogger(__name__)


@dataclass
class PricingResult:
    """Result of Monte Carlo pricing for one player/game/line."""

    player_id: str
    game_id: str
    line: float
    proj_minutes: float
    proj_potential_assists: float
    proj_assists: float
    fair_over_prob: float
    fair_under_prob: float
    fair_over_price: float
    fair_under_price: float
    mean_assists: float
    median_assists: float
    std_assists: float
    p10_assists: float
    p90_assists: float
    sim_draws: int

    @property
    def push_prob(self) -> float:
        return 1.0 - self.fair_over_prob - self.fair_under_prob


class MonteCarloPricer:
    """Simulate assists via three-layer model and price over/under."""

    def __init__(self, n_draws: int = 10_000, seed: int = 42):
        self.n_draws = n_draws
        self.rng = np.random.RandomState(seed)

    def price(
        self,
        player_id: str,
        game_id: str,
        line: float,
        proj_minutes_mean: float,
        proj_minutes_std: float | None,
        proj_opportunity_rate: float,
        proj_opportunity_std: float | None,
        proj_conversion_rate: float,
        proj_conversion_std: float | None,
    ) -> PricingResult:
        """Run Monte Carlo simulation for one player/game/line.

        Flow:
        1. Sample minutes from truncated normal
        2. Sample opportunity rate (potential assists per minute)
        3. Compute potential assists = minutes * opportunity_rate
        4. Sample conversion rate
        5. Compute assists = potential_assists * conversion_rate
        6. Round to integer (NBA assists are integers)
        """
        # Defaults for std if not provided
        min_std = proj_minutes_std if proj_minutes_std and proj_minutes_std > 0 else 5.0
        opp_std = proj_opportunity_std if proj_opportunity_std and proj_opportunity_std > 0 else 0.05
        conv_std = proj_conversion_std if proj_conversion_std and proj_conversion_std > 0 else 0.05

        # Sample minutes (truncated at 0 and 48)
        minutes_samples = self.rng.normal(proj_minutes_mean, min_std, self.n_draws)
        minutes_samples = np.clip(minutes_samples, 0, 48)

        # Sample opportunity rate (potential assists per minute)
        opp_samples = self.rng.normal(proj_opportunity_rate, opp_std, self.n_draws)
        opp_samples = np.clip(opp_samples, 0, None)

        # Potential assists
        potential_assists = minutes_samples * opp_samples

        # Sample conversion rate (bounded 0-1)
        conv_samples = self.rng.normal(proj_conversion_rate, conv_std, self.n_draws)
        conv_samples = np.clip(conv_samples, 0, 1)

        # Assists
        assists_raw = potential_assists * conv_samples

        # Sanity check: no negative assists, cap at reasonable max
        assists_raw = np.clip(assists_raw, 0, 30)

        # For over/under, we compare against the line
        # Standard: over = strictly greater than line, under = strictly less
        # Push = exactly equal
        over_count = np.sum(assists_raw > line)
        under_count = np.sum(assists_raw < line)
        push_count = np.sum(np.isclose(assists_raw, line, atol=0.25))

        # Probabilities (exclude pushes, as is standard)
        non_push = over_count + under_count
        if non_push > 0:
            fair_over_prob = over_count / self.n_draws
            fair_under_prob = under_count / self.n_draws
        else:
            fair_over_prob = 0.5
            fair_under_prob = 0.5

        # Normalize to sum to 1 (excluding push)
        total = fair_over_prob + fair_under_prob
        if total > 0:
            fair_over_prob = fair_over_prob / total
            fair_under_prob = fair_under_prob / total

        # Convert to American odds
        try:
            fair_over_price = implied_to_american(fair_over_prob)
        except ValueError:
            fair_over_price = 0.0
        try:
            fair_under_price = implied_to_american(fair_under_prob)
        except ValueError:
            fair_under_price = 0.0

        return PricingResult(
            player_id=player_id,
            game_id=game_id,
            line=line,
            proj_minutes=proj_minutes_mean,
            proj_potential_assists=float(np.mean(potential_assists)),
            proj_assists=float(np.mean(assists_raw)),
            fair_over_prob=fair_over_prob,
            fair_under_prob=fair_under_prob,
            fair_over_price=fair_over_price,
            fair_under_price=fair_under_price,
            mean_assists=float(np.mean(assists_raw)),
            median_assists=float(np.median(assists_raw)),
            std_assists=float(np.std(assists_raw)),
            p10_assists=float(np.percentile(assists_raw, 10)),
            p90_assists=float(np.percentile(assists_raw, 90)),
            sim_draws=self.n_draws,
        )

    def price_batch(
        self,
        players: list[dict],
    ) -> list[PricingResult]:
        """Price a batch of player/game/line combinations.

        Each dict in players should contain:
        - player_id, game_id, line
        - proj_minutes_mean, proj_minutes_std
        - proj_opportunity_rate, proj_opportunity_std
        - proj_conversion_rate, proj_conversion_std
        """
        results = []
        for p in players:
            try:
                result = self.price(
                    player_id=p["player_id"],
                    game_id=p["game_id"],
                    line=p["line"],
                    proj_minutes_mean=p["proj_minutes_mean"],
                    proj_minutes_std=p.get("proj_minutes_std"),
                    proj_opportunity_rate=p["proj_opportunity_rate"],
                    proj_opportunity_std=p.get("proj_opportunity_std"),
                    proj_conversion_rate=p["proj_conversion_rate"],
                    proj_conversion_std=p.get("proj_conversion_std"),
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Pricing failed for {p.get('player_id')}: {e}")
        return results
