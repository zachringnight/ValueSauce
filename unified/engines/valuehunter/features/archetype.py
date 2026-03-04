"""Player archetype classifier for NBA 3PM Props Engine.

Classifies players into one of five archetypes based on shot-mix,
role, and tracking data.  When tracking data is unavailable the
classifier falls back to a boxscore-only heuristic.

Archetypes
----------
- movement_wing_shooter : High catch-and-shoot share, quick touches,
  heavily assisted on threes.
- pull_up_guard : Self-creating guard with high pull-up share and
  usage.
- stretch_big : Center / power-forward who spaces the floor with low
  touch volume.
- stationary_spacer : Minimal self-creation, very high assisted share,
  low usage.
- bench_microwave : Low-minute reserve with volatile three-point
  attempt rates.
"""

from __future__ import annotations

import logging
import statistics
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants – thresholds taken from the design spec
# ---------------------------------------------------------------------------

# movement_wing_shooter
_MWS_CATCH_SHOOT_SHARE_MIN = 0.55
_MWS_AVG_SEC_PER_TOUCH_MAX = 3.0
_MWS_ASSISTED_3PM_SHARE_MIN = 0.70

# pull_up_guard
_PUG_PULL_UP_SHARE_MIN = 0.35
_PUG_USAGE_MIN = 0.22
_PUG_AVG_DRIBBLES_MIN = 2.5

# stretch_big
_SB_POSITIONS = {"C", "PF"}
_SB_TOUCHES_PER_MIN_MAX = 3.5

# stationary_spacer
_SS_PULL_UP_SHARE_MAX = 0.15
_SS_ASSISTED_3PM_SHARE_MIN = 0.80
_SS_USAGE_MAX = 0.18

# bench_microwave
_BM_MINUTES_AVG_MAX = 24.0
_BM_STARTER_PCT_MAX = 0.30


class ArchetypeClassifier:
    """Classify a player into one of five strategic archetypes."""

    # The ordered list of archetypes to attempt (first match wins).
    ARCHETYPES = [
        "movement_wing_shooter",
        "pull_up_guard",
        "stretch_big",
        "stationary_spacer",
        "bench_microwave",
    ]
    DEFAULT_ARCHETYPE = "stationary_spacer"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        player_games: list[dict[str, Any]],
        tracking_games: list[dict[str, Any]] | None = None,
    ) -> str:
        """Return the archetype string for a player.

        Parameters
        ----------
        player_games:
            List of boxscore-level game dicts. Expected keys include
            ``minutes``, ``started``, ``fg3a``, ``fg3m``, ``fga``,
            ``usage_rate``, ``position``, etc.
        tracking_games:
            Optional list of tracking-level game dicts with keys like
            ``catch_shoot_3pa``, ``pull_up_3pa``, ``assisted_3pm``,
            ``touches``, ``time_of_poss``, ``avg_seconds_per_touch``,
            ``avg_dribbles_per_touch``, etc.

        Returns
        -------
        str
            One of the five archetype labels.
        """
        if not player_games:
            logger.warning("No player games supplied – returning default archetype.")
            return self.DEFAULT_ARCHETYPE

        role_features = self._compute_role_features(player_games)

        if tracking_games:
            shot_mix = self._compute_shot_mix_features(tracking_games)
        else:
            shot_mix = self._estimate_shot_mix_from_boxscore(player_games)

        archetype = self._apply_rules(role_features, shot_mix)
        logger.info("Classified player as archetype=%s", archetype)
        return archetype

    # ------------------------------------------------------------------
    # Feature computation helpers
    # ------------------------------------------------------------------

    def _compute_shot_mix_features(
        self, tracking_games: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Derive shot-mix features from tracking data.

        Returns a dict with keys:
        - catch_shoot_share
        - pull_up_share
        - assisted_3pm_share
        - avg_seconds_per_touch
        - avg_dribbles_per_touch
        - touches_per_min
        """
        total_3pa = 0.0
        total_cs_3pa = 0.0
        total_pu_3pa = 0.0
        total_3pm = 0.0
        total_assisted_3pm = 0.0
        total_touches = 0.0
        total_minutes = 0.0
        sec_per_touch_vals: list[float] = []
        drib_per_touch_vals: list[float] = []

        for g in tracking_games:
            fg3a = float(g.get("fg3a", 0) or 0)
            cs_3pa = float(g.get("catch_shoot_3pa", 0) or 0)
            pu_3pa = float(g.get("pull_up_3pa", 0) or 0)
            fg3m = float(g.get("fg3m", 0) or 0)
            assisted_3pm = float(g.get("assisted_3pm", 0) or 0)
            touches = float(g.get("touches", 0) or 0)
            minutes = float(g.get("minutes", 0) or 0)
            spt = g.get("avg_seconds_per_touch")
            dpt = g.get("avg_dribbles_per_touch")

            total_3pa += fg3a
            total_cs_3pa += cs_3pa
            total_pu_3pa += pu_3pa
            total_3pm += fg3m
            total_assisted_3pm += assisted_3pm
            total_touches += touches
            total_minutes += minutes

            if spt is not None:
                sec_per_touch_vals.append(float(spt))
            if dpt is not None:
                drib_per_touch_vals.append(float(dpt))

        catch_shoot_share = (total_cs_3pa / total_3pa) if total_3pa > 0 else 0.0
        pull_up_share = (total_pu_3pa / total_3pa) if total_3pa > 0 else 0.0
        assisted_3pm_share = (
            (total_assisted_3pm / total_3pm) if total_3pm > 0 else 0.0
        )
        avg_seconds_per_touch = (
            statistics.mean(sec_per_touch_vals) if sec_per_touch_vals else 0.0
        )
        avg_dribbles_per_touch = (
            statistics.mean(drib_per_touch_vals) if drib_per_touch_vals else 0.0
        )
        touches_per_min = (total_touches / total_minutes) if total_minutes > 0 else 0.0

        return {
            "catch_shoot_share": catch_shoot_share,
            "pull_up_share": pull_up_share,
            "assisted_3pm_share": assisted_3pm_share,
            "avg_seconds_per_touch": avg_seconds_per_touch,
            "avg_dribbles_per_touch": avg_dribbles_per_touch,
            "touches_per_min": touches_per_min,
        }

    def _compute_role_features(
        self, player_games: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Derive role features from boxscore data.

        Returns a dict with keys:
        - minutes_avg
        - starter_pct
        - usage_rate_avg
        - position (most common position string)
        - fg3a_rate_std  (std-dev of 3PA per game – proxy for volatility)
        """
        minutes_vals: list[float] = []
        started_vals: list[float] = []
        usage_vals: list[float] = []
        fg3a_vals: list[float] = []
        position_counts: dict[str, int] = {}

        for g in player_games:
            minutes = float(g.get("minutes", 0) or 0)
            started = 1.0 if g.get("started") else 0.0
            usage = float(g.get("usage_rate", 0) or 0)
            fg3a = float(g.get("fg3a", 0) or 0)
            pos = g.get("position", "")

            minutes_vals.append(minutes)
            started_vals.append(started)
            usage_vals.append(usage)
            fg3a_vals.append(fg3a)

            if pos:
                position_counts[pos] = position_counts.get(pos, 0) + 1

        n = len(player_games)
        minutes_avg = statistics.mean(minutes_vals) if minutes_vals else 0.0
        starter_pct = (sum(started_vals) / n) if n > 0 else 0.0
        usage_rate_avg = statistics.mean(usage_vals) if usage_vals else 0.0
        fg3a_rate_std = statistics.pstdev(fg3a_vals) if len(fg3a_vals) >= 2 else 0.0
        position = (
            max(position_counts, key=position_counts.get)
            if position_counts
            else ""
        )

        return {
            "minutes_avg": minutes_avg,
            "starter_pct": starter_pct,
            "usage_rate_avg": usage_rate_avg,
            "position": position,
            "fg3a_rate_std": fg3a_rate_std,
        }

    def _estimate_shot_mix_from_boxscore(
        self, player_games: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Fallback when tracking data is unavailable.

        Produces *proxy* values from plain boxscore stats.  These are
        deliberately conservative so the classifier can still make
        reasonable decisions.
        """
        total_fg3a = 0.0
        total_fg3m = 0.0
        total_fga = 0.0
        total_minutes = 0.0
        total_assists = 0.0

        for g in player_games:
            total_fg3a += float(g.get("fg3a", 0) or 0)
            total_fg3m += float(g.get("fg3m", 0) or 0)
            total_fga += float(g.get("fga", 0) or 0)
            total_minutes += float(g.get("minutes", 0) or 0)
            total_assists += float(g.get("assists", 0) or 0)

        fg3_rate = (total_fg3a / total_fga) if total_fga > 0 else 0.0
        assists_per_min = (total_assists / total_minutes) if total_minutes > 0 else 0.0

        # Heuristic: if a player has a high 3pt rate and low assist rate,
        # they are likely a catch-and-shoot / spacer type.
        # High assist rate + reasonable 3pt volume => pull-up guard proxy.
        estimated_catch_shoot_share = max(0.0, min(1.0, fg3_rate * 1.5))
        estimated_pull_up_share = max(
            0.0, min(1.0, assists_per_min * 0.25)
        )
        # Without tracking we cannot know assisted share so assume moderate.
        estimated_assisted_share = 0.60 if fg3_rate > 0.30 else 0.50

        return {
            "catch_shoot_share": estimated_catch_shoot_share,
            "pull_up_share": estimated_pull_up_share,
            "assisted_3pm_share": estimated_assisted_share,
            "avg_seconds_per_touch": 0.0,  # unknown
            "avg_dribbles_per_touch": 0.0,  # unknown
            "touches_per_min": 0.0,  # unknown
        }

    # ------------------------------------------------------------------
    # Rule engine
    # ------------------------------------------------------------------

    def _apply_rules(
        self,
        role: dict[str, Any],
        shot_mix: dict[str, float],
    ) -> str:
        """Apply archetype classification rules in priority order.

        First matching archetype wins.
        """
        # 1. movement_wing_shooter
        if (
            shot_mix["catch_shoot_share"] > _MWS_CATCH_SHOOT_SHARE_MIN
            and shot_mix["avg_seconds_per_touch"] < _MWS_AVG_SEC_PER_TOUCH_MAX
            and shot_mix["assisted_3pm_share"] > _MWS_ASSISTED_3PM_SHARE_MIN
        ):
            # avg_seconds_per_touch == 0 means tracking unavailable; skip
            # this rule only when we positively know the touch time is low.
            if shot_mix["avg_seconds_per_touch"] > 0:
                return "movement_wing_shooter"

        # 2. pull_up_guard
        if (
            shot_mix["pull_up_share"] > _PUG_PULL_UP_SHARE_MIN
            and role["usage_rate_avg"] > _PUG_USAGE_MIN
            and shot_mix["avg_dribbles_per_touch"] > _PUG_AVG_DRIBBLES_MIN
        ):
            return "pull_up_guard"

        # 3. stretch_big (position-gated)
        if (
            role["position"] in _SB_POSITIONS
            and shot_mix["touches_per_min"] < _SB_TOUCHES_PER_MIN_MAX
        ):
            # Accept if touches_per_min is 0 (unknown) only when the
            # position is definitely C or PF.
            return "stretch_big"

        # 4. stationary_spacer
        if (
            shot_mix["pull_up_share"] < _SS_PULL_UP_SHARE_MAX
            and shot_mix["assisted_3pm_share"] > _SS_ASSISTED_3PM_SHARE_MIN
            and role["usage_rate_avg"] < _SS_USAGE_MAX
        ):
            return "stationary_spacer"

        # 5. bench_microwave
        if (
            role["minutes_avg"] < _BM_MINUTES_AVG_MAX
            and role["starter_pct"] < _BM_STARTER_PCT_MAX
        ):
            return "bench_microwave"

        # Fallback – if no rule matches, infer from most distinguishing
        # single feature.
        return self._fallback_classify(role, shot_mix)

    def _fallback_classify(
        self,
        role: dict[str, Any],
        shot_mix: dict[str, float],
    ) -> str:
        """Last-resort classification when no explicit rule fires."""
        if role["position"] in _SB_POSITIONS:
            return "stretch_big"
        if role["minutes_avg"] < _BM_MINUTES_AVG_MAX:
            return "bench_microwave"
        if shot_mix["pull_up_share"] > 0.25:
            return "pull_up_guard"
        if shot_mix["catch_shoot_share"] > 0.45:
            return "movement_wing_shooter"
        return self.DEFAULT_ARCHETYPE
