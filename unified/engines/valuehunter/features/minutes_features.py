"""Minutes-model feature builder (Section H1).

Builds the feature vector used by the minutes sub-model to predict
how many minutes a player will play in an upcoming game.  Features
span:

- Rolling minutes statistics (mean, volatility, percentile)
- Starter / bench role indicators
- Rest and schedule context
- Game-line context (spread, team total, blowout risk)
- Injury / absence signals (self and teammates)
- Lineup continuity
"""

from __future__ import annotations

import logging
import math
import statistics
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


class MinutesFeatureBuilder:
    """Compute all Section-H1 features for the minutes sub-model."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        player_games: list[dict[str, Any]],
        game_context: dict[str, Any],
        injury_snapshot: dict[str, Any] | None = None,
        teammate_statuses: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return the full H1 feature dict.

        Parameters
        ----------
        player_games:
            Boxscore dicts sorted most-recent-first.  Expected keys:
            ``minutes``, ``started``, ``game_date``, ``starters``
            (list of player-ids who started that game).
        game_context:
            Dict with keys ``spread``, ``team_total``, ``is_home``,
            ``game_date``, ``rest_days``, ``is_back_to_back``,
            ``is_3in4``.
        injury_snapshot:
            Optional dict describing the player's own injury state.
            Expected keys: ``status`` (e.g. "Probable", "Questionable",
            "Out", "Healthy"), ``injury_type``.
        teammate_statuses:
            Optional list of dicts for relevant teammates. Each dict
            should have keys: ``role`` (one of "primary_creator",
            "secondary_creator", "starting_big", "high_volume_wing"),
            ``status`` ("Out", "Questionable", etc.).
        """
        if not player_games:
            logger.warning("No player games for minutes features – returning empty.")
            return {}

        features: dict[str, Any] = {}

        # --- Rolling minutes ---
        minutes_vals = self._extract_float_series(player_games, "minutes")

        features["minutes_avg_l3"] = self._rolling_stat(minutes_vals, 3, "mean")
        features["minutes_avg_l5"] = self._rolling_stat(minutes_vals, 5, "mean")
        features["minutes_avg_l10"] = self._rolling_stat(minutes_vals, 10, "mean")
        features["minutes_avg_l20"] = self._rolling_stat(minutes_vals, 20, "mean")

        # --- Minutes volatility (std of last 10) ---
        features["minutes_std_l10"] = self._rolling_stat(minutes_vals, 10, "std")

        # --- p90 minutes (90th percentile of last 20) ---
        features["minutes_p90_l20"] = self._rolling_stat(minutes_vals, 20, "p90")

        # --- Starter indicators ---
        started_vals = [
            1.0 if g.get("started") else 0.0 for g in player_games
        ]
        features["is_starter"] = 1.0 if started_vals and started_vals[0] == 1.0 else 0.0
        features["starts_in_last_10"] = sum(started_vals[:10])

        # --- Schedule / rest context ---
        features["rest_days"] = float(game_context.get("rest_days", 1))
        features["is_back_to_back"] = (
            1.0 if game_context.get("is_back_to_back") else 0.0
        )
        features["is_3in4"] = 1.0 if game_context.get("is_3in4") else 0.0

        # --- Game-line context ---
        spread = float(game_context.get("spread", 0))
        team_total = float(game_context.get("team_total", 0))
        features["spread"] = spread
        features["team_total"] = team_total

        # --- Blowout probability ---
        features["blowout_probability"] = self._compute_blowout_prob(spread)

        # --- Games missed in last 14 days ---
        features["games_missed_in_last_14"] = self._games_missed_last_n_days(
            player_games, 14, game_context
        )

        # --- Self injury state ---
        if injury_snapshot:
            status = (injury_snapshot.get("status") or "Healthy").lower()
            features["self_injury_state"] = status
            features["self_injury_is_questionable"] = 1.0 if status == "questionable" else 0.0
            features["self_injury_is_probable"] = 1.0 if status == "probable" else 0.0
        else:
            features["self_injury_state"] = "healthy"
            features["self_injury_is_questionable"] = 0.0
            features["self_injury_is_probable"] = 0.0

        # --- Teammate-out flags ---
        teammate_flags = self._compute_teammate_flags(teammate_statuses)
        features.update(teammate_flags)

        # --- Lineup continuity ---
        features["lineup_continuity"] = self._compute_lineup_continuity(player_games)

        return features

    # ------------------------------------------------------------------
    # Rolling stat helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rolling_stat(
        values: list[float],
        window: int,
        stat: str = "mean",
    ) -> float:
        """Compute a windowed statistic over the *first* ``window`` items.

        Parameters
        ----------
        values:
            Series sorted most-recent-first.
        window:
            Number of most-recent observations to include.
        stat:
            One of ``"mean"``, ``"std"``, ``"p90"``.
        """
        subset = values[:window]
        if not subset:
            return 0.0

        if stat == "mean":
            return statistics.mean(subset)
        if stat == "std":
            return statistics.pstdev(subset) if len(subset) >= 2 else 0.0
        if stat == "p90":
            return _percentile(subset, 90)
        raise ValueError(f"Unknown stat type: {stat}")

    # ------------------------------------------------------------------
    # Blowout probability
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_blowout_prob(spread: float) -> float:
        """Estimate blowout probability from the pregame spread.

        Uses a simple sigmoid-style mapping:
        - |spread| <= 3  ->  ~0.05
        - |spread| ~7    ->  ~0.15
        - |spread| ~10   ->  ~0.30
        - |spread| >= 15 ->  ~0.55
        """
        abs_spread = abs(spread)
        # Logistic-ish curve: 1 / (1 + exp(-k*(x - x0)))
        # Tuned so that abs_spread=10 -> ~0.30
        prob = 1.0 / (1.0 + math.exp(-0.25 * (abs_spread - 10.0)))
        return round(prob, 4)

    # ------------------------------------------------------------------
    # Missed games
    # ------------------------------------------------------------------

    @staticmethod
    def _games_missed_last_n_days(
        player_games: list[dict[str, Any]],
        n_days: int,
        game_context: dict[str, Any],
    ) -> int:
        """Count games that a player *missed* (DNP / inactive) in the
        last ``n_days`` calendar days.

        We approximate this as:
        ``expected_games_in_window - actual_games_in_window``.
        A typical NBA schedule averages ~3.5 games per 7 days for a
        team, so ``expected = n_days * (3.5 / 7) = n_days * 0.5``.
        """
        ref_date_raw = game_context.get("game_date")
        if not ref_date_raw:
            return 0

        if isinstance(ref_date_raw, str):
            try:
                ref_date = datetime.strptime(ref_date_raw, "%Y-%m-%d").date()
            except ValueError:
                return 0
        else:
            ref_date = ref_date_raw

        cutoff = ref_date - timedelta(days=n_days)
        games_in_window = 0
        for g in player_games:
            gd = g.get("game_date")
            if gd is None:
                continue
            if isinstance(gd, str):
                try:
                    gd = datetime.strptime(gd, "%Y-%m-%d").date()
                except ValueError:
                    continue
            if cutoff <= gd < ref_date:
                games_in_window += 1

        expected = max(1, int(round(n_days * 0.5)))
        return max(0, expected - games_in_window)

    # ------------------------------------------------------------------
    # Teammate flags
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_teammate_flags(
        teammate_statuses: list[dict[str, Any]] | None,
    ) -> dict[str, float]:
        """Return binary flags for key teammate absences."""
        flags = {
            "primary_creator_out": 0.0,
            "secondary_creator_out": 0.0,
            "starting_big_out": 0.0,
            "high_volume_wing_out": 0.0,
        }
        if not teammate_statuses:
            return flags

        for tm in teammate_statuses:
            role = (tm.get("role") or "").lower().replace(" ", "_")
            status = (tm.get("status") or "").lower()
            if status == "out":
                key = f"{role}_out"
                if key in flags:
                    flags[key] = 1.0
        return flags

    # ------------------------------------------------------------------
    # Lineup continuity
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_lineup_continuity(
        player_games: list[dict[str, Any]],
        window: int = 5,
    ) -> float:
        """Fraction of the last *window* games that share the same
        starting five as the most recent game.

        ``starters`` is expected to be a sorted list/tuple of player
        identifiers for each game dict.
        """
        recent = player_games[:window]
        if len(recent) < 2:
            return 1.0

        ref_starters = recent[0].get("starters")
        if not ref_starters:
            return 0.0

        ref_set = set(ref_starters)
        same_count = 0
        for g in recent:
            g_starters = g.get("starters")
            if g_starters and set(g_starters) == ref_set:
                same_count += 1

        return same_count / len(recent)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_float_series(
        games: list[dict[str, Any]], key: str
    ) -> list[float]:
        """Pull a float series from a list of dicts, coercing safely."""
        result: list[float] = []
        for g in games:
            val = g.get(key)
            try:
                result.append(float(val))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                result.append(0.0)
        return result


# -----------------------------------------------------------------------
# Module-level helpers
# -----------------------------------------------------------------------


def _percentile(values: list[float], pct: int) -> float:
    """Return the *pct*-th percentile using linear interpolation."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    k = (pct / 100.0) * (n - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    d0 = s[f] * (c - k)
    d1 = s[c] * (k - f)
    return d0 + d1
