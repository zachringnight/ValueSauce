"""Make-rate feature builder (Section H4).

Produces the feature vector for the 3-point *make-rate* sub-model,
which predicts ``P(make | attempt)`` – i.e. the shooting percentage
conditioned on a three-point attempt being taken.

Key ideas:
- Rolling shooting percentages at multiple windows.
- Empirical-Bayes shrinkage toward archetype / league-mean priors so
  that small-sample players aren't given extreme rates.
- Catch-shoot vs. pull-up split rates (when tracking is available).
- Opponent closest-defender environment as a defensive context feature.
- Creator-availability flags that proxy for shot-quality shifts when
  a primary playmaker is absent.
"""

from __future__ import annotations

import logging
import statistics
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# League-wide prior constants (2023-24 season reference values)
# ---------------------------------------------------------------------------
LEAGUE_3PT_PCT: float = 0.363
CATCH_SHOOT_LEAGUE_AVG: float = 0.374
PULL_UP_LEAGUE_AVG: float = 0.329


class MakeRateFeatureBuilder:
    """Compute Section-H4 features for the make-rate sub-model."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        player_games: list[dict[str, Any]],
        tracking_games: list[dict[str, Any]] | None,
        opponent_shooting: list[dict[str, Any]] | None,
        game_context: dict[str, Any],
        archetype: str,
    ) -> dict[str, Any]:
        """Return the full H4 feature dict.

        Parameters
        ----------
        player_games:
            Boxscore dicts sorted most-recent-first.
        tracking_games:
            Tracking-level game dicts (may be ``None`` or empty).
        opponent_shooting:
            Opponent defensive shooting context.
        game_context:
            Pregame context dict.
        archetype:
            Player archetype label.
        """
        if not player_games:
            logger.warning("No player games – returning empty make-rate features.")
            return {}

        features: dict[str, Any] = {}
        has_tracking = bool(tracking_games)
        features["tracking_available"] = has_tracking

        # --- Rolling FG3% from boxscore ---
        features.update(self._rolling_fg3_pct(player_games))

        # --- Tracking-based split rates ---
        if has_tracking:
            features.update(self._tracking_split_rates(tracking_games))  # type: ignore[arg-type]
        else:
            features["rolling_catch_shoot_fg3_pct"] = None
            features["rolling_pull_up_fg3_pct"] = None

        # --- Assisted / unassisted share ---
        assisted_share, unassisted_share = self._compute_assisted_shares(
            tracking_games if has_tracking else None,
            player_games,
        )
        features["assisted_3pm_share"] = assisted_share
        features["unassisted_3pm_share"] = unassisted_share

        # --- Projected shot-type shares (for expected make-rate) ---
        if has_tracking:
            proj_cs, proj_pu = self._projected_shares(tracking_games)  # type: ignore[arg-type]
        else:
            proj_cs, proj_pu = 0.5, 0.5  # uninformative prior
        features["projected_catch_shoot_share"] = proj_cs
        features["projected_pull_up_share"] = proj_pu

        # --- Creator availability flags ---
        features.update(self._creator_availability(game_context))

        # --- Team spacing proxy ---
        features["team_spacing_proxy"] = self._team_spacing_proxy(player_games)

        # --- Opponent closest-defender context ---
        features["opp_closest_defender_context"] = self._opp_defender_context(
            opponent_shooting
        )

        # --- Game context ---
        features["is_home"] = 1.0 if game_context.get("is_home") else 0.0
        features["rest_days"] = float(game_context.get("rest_days", 1))
        features["archetype"] = archetype

        # --- Empirical-Bayes shrunk FG3% ---
        total_3pa = sum(float(g.get("fg3a", 0) or 0) for g in player_games)
        total_3pm = sum(float(g.get("fg3m", 0) or 0) for g in player_games)
        archetype_prior = self._archetype_prior(archetype)
        features["empirical_bayes_fg3_pct"] = self._empirical_bayes_shrink(
            player_3pa=total_3pa,
            player_3pm=total_3pm,
            prior_rate=archetype_prior,
        )

        # Shrunk catch-shoot rate (tracking only)
        if has_tracking:
            cs_3pa = sum(
                float(g.get("catch_shoot_3pa", 0) or 0) for g in tracking_games  # type: ignore[union-attr]
            )
            cs_3pm = sum(
                float(g.get("catch_shoot_3pm", 0) or 0) for g in tracking_games  # type: ignore[union-attr]
            )
            features["eb_catch_shoot_fg3_pct"] = self._empirical_bayes_shrink(
                player_3pa=cs_3pa,
                player_3pm=cs_3pm,
                prior_rate=CATCH_SHOOT_LEAGUE_AVG,
            )
            pu_3pa = sum(
                float(g.get("pull_up_3pa", 0) or 0) for g in tracking_games  # type: ignore[union-attr]
            )
            pu_3pm = sum(
                float(g.get("pull_up_3pm", 0) or 0) for g in tracking_games  # type: ignore[union-attr]
            )
            features["eb_pull_up_fg3_pct"] = self._empirical_bayes_shrink(
                player_3pa=pu_3pa,
                player_3pm=pu_3pm,
                prior_rate=PULL_UP_LEAGUE_AVG,
            )
        else:
            features["eb_catch_shoot_fg3_pct"] = None
            features["eb_pull_up_fg3_pct"] = None

        return features

    # ------------------------------------------------------------------
    # Rolling FG3% from boxscore
    # ------------------------------------------------------------------

    @staticmethod
    def _rolling_fg3_pct(
        player_games: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Compute rolling 3-point percentage at several windows."""
        result: dict[str, float] = {}

        for label, window in [("l10", 10), ("l20", 20), ("season", None)]:
            subset = player_games[:window] if window else player_games
            fg3a = sum(float(g.get("fg3a", 0) or 0) for g in subset)
            fg3m = sum(float(g.get("fg3m", 0) or 0) for g in subset)
            result[f"rolling_fg3_pct_{label}"] = (fg3m / fg3a) if fg3a > 0 else 0.0

        # Overall (same as season here but kept explicit)
        total_3pa = sum(float(g.get("fg3a", 0) or 0) for g in player_games)
        total_3pm = sum(float(g.get("fg3m", 0) or 0) for g in player_games)
        result["rolling_fg3_pct_overall"] = (
            (total_3pm / total_3pa) if total_3pa > 0 else 0.0
        )
        return result

    # ------------------------------------------------------------------
    # Tracking-based split rates
    # ------------------------------------------------------------------

    @staticmethod
    def _tracking_split_rates(
        tracking_games: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Rolling catch-shoot and pull-up FG3% from tracking data."""
        cs_3pa = sum(float(g.get("catch_shoot_3pa", 0) or 0) for g in tracking_games)
        cs_3pm = sum(float(g.get("catch_shoot_3pm", 0) or 0) for g in tracking_games)
        pu_3pa = sum(float(g.get("pull_up_3pa", 0) or 0) for g in tracking_games)
        pu_3pm = sum(float(g.get("pull_up_3pm", 0) or 0) for g in tracking_games)

        return {
            "rolling_catch_shoot_fg3_pct": (cs_3pm / cs_3pa) if cs_3pa > 0 else 0.0,
            "rolling_pull_up_fg3_pct": (pu_3pm / pu_3pa) if pu_3pa > 0 else 0.0,
        }

    # ------------------------------------------------------------------
    # Assisted share
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_assisted_shares(
        tracking_games: list[dict[str, Any]] | None,
        player_games: list[dict[str, Any]],
    ) -> tuple[float, float]:
        """Return ``(assisted_3pm_share, unassisted_3pm_share)``."""
        if tracking_games:
            total_3pm = sum(
                float(g.get("fg3m", 0) or 0) for g in tracking_games
            )
            assisted = sum(
                float(g.get("assisted_3pm", 0) or 0) for g in tracking_games
            )
            if total_3pm > 0:
                ast_share = assisted / total_3pm
                return ast_share, 1.0 - ast_share

        # Fallback: assume moderate assisted share
        return 0.60, 0.40

    # ------------------------------------------------------------------
    # Projected shot-type shares
    # ------------------------------------------------------------------

    @staticmethod
    def _projected_shares(
        tracking_games: list[dict[str, Any]],
        window: int = 10,
    ) -> tuple[float, float]:
        """Estimate projected catch-shoot and pull-up shares going
        forward, using the most recent *window* tracking games."""
        subset = tracking_games[:window]
        total_3pa = sum(float(g.get("fg3a", 0) or 0) for g in subset)
        cs = sum(float(g.get("catch_shoot_3pa", 0) or 0) for g in subset)
        pu = sum(float(g.get("pull_up_3pa", 0) or 0) for g in subset)

        if total_3pa <= 0:
            return 0.5, 0.5
        return cs / total_3pa, pu / total_3pa

    # ------------------------------------------------------------------
    # Creator availability
    # ------------------------------------------------------------------

    @staticmethod
    def _creator_availability(game_context: dict[str, Any]) -> dict[str, float]:
        """Extract creator-availability boolean flags from game context.

        The game_context may embed ``teammate_statuses`` or explicit
        flags like ``primary_creator_available``.
        """
        result: dict[str, float] = {}
        for role in [
            "primary_creator_available",
            "secondary_creator_available",
        ]:
            val = game_context.get(role)
            if val is not None:
                result[role] = 1.0 if val else 0.0
            else:
                result[role] = 1.0  # assume available by default
        return result

    # ------------------------------------------------------------------
    # Team spacing proxy
    # ------------------------------------------------------------------

    @staticmethod
    def _team_spacing_proxy(player_games: list[dict[str, Any]]) -> float:
        """Proxy for team spacing: team 3PA rate over last 10 games.

        Uses ``team_fga`` and ``team_fg3a`` when available; otherwise
        returns 0.36 (league average).
        """
        subset = player_games[:10]
        team_fg3a = sum(float(g.get("team_fg3a", 0) or 0) for g in subset)
        team_fga = sum(float(g.get("team_fga", 0) or 0) for g in subset)
        if team_fga > 0:
            return team_fg3a / team_fga
        return 0.36  # league-average fallback

    # ------------------------------------------------------------------
    # Opponent defender context
    # ------------------------------------------------------------------

    @staticmethod
    def _opp_defender_context(
        opponent_shooting: list[dict[str, Any]] | None,
    ) -> float:
        """Return the average closest-defender distance for the
        upcoming opponent.  Higher values => more open looks."""
        if not opponent_shooting:
            return 0.0
        vals = [
            float(rec.get("opp_closest_def_dist", 0) or 0)
            for rec in opponent_shooting
        ]
        return statistics.mean(vals) if vals else 0.0

    # ------------------------------------------------------------------
    # Empirical-Bayes shrinkage
    # ------------------------------------------------------------------

    @staticmethod
    def _empirical_bayes_shrink(
        player_3pa: float,
        player_3pm: float,
        prior_rate: float,
        prior_weight: float = 100.0,
    ) -> float:
        """Shrink observed shooting percentage toward a prior.

        Formula (beta-binomial conjugate update):

            shrunk = (player_3pm + prior_weight * prior_rate)
                   / (player_3pa + prior_weight)

        When ``player_3pa`` is very small the estimate is pulled toward
        ``prior_rate``; as sample size grows the player's own rate
        dominates.

        Parameters
        ----------
        player_3pa:
            Total three-point attempts (sample size).
        player_3pm:
            Total three-point makes.
        prior_rate:
            The prior mean (e.g. league average, archetype average).
        prior_weight:
            Effective prior sample size (higher = more shrinkage).
        """
        return (player_3pm + prior_weight * prior_rate) / (
            player_3pa + prior_weight
        )

    # ------------------------------------------------------------------
    # Archetype-aware prior
    # ------------------------------------------------------------------

    @staticmethod
    def _archetype_prior(archetype: str) -> float:
        """Return a reasonable prior 3PT% for a given archetype."""
        priors = {
            "movement_wing_shooter": 0.380,
            "pull_up_guard": 0.345,
            "stretch_big": 0.355,
            "stationary_spacer": 0.375,
            "bench_microwave": 0.360,
        }
        return priors.get(archetype, LEAGUE_3PT_PCT)
