"""Decision engine for NBA 3PM Props betting (Section J).

Evaluates Monte Carlo simulation outputs against market prices, applies
exclusion and trigger rules, sizes positions, and enforces correlation
limits before emitting bet decisions.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Optional

from ..config.settings import Settings, get_settings
from .monte_carlo import SimulationResult

logger = logging.getLogger(__name__)


class DecisionEngine:
    """Converts simulation results + market prices into actionable bet decisions.

    Workflow per opportunity:
    1. Convert book prices to raw implied probabilities.
    2. Remove vig using the power method.
    3. Compare model fair probability to market no-vig probability.
    4. Compute expected value for over and under.
    5. Determine recommended side.
    6. Size the stake (flat or quarter-Kelly).
    7. Apply exclusion rules (data quality, context).
    8. Apply trigger rules (EV, edge, hold, staleness).
    9. Return full decision dict matching ``bet_decisions`` schema.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

        # Decision thresholds
        self.min_ev_pct: float = self.settings.min_ev_pct
        self.min_edge_prob_pts: float = self.settings.min_edge_prob_pts
        self.max_stale_odds_minutes: int = self.settings.max_stale_odds_minutes
        self.max_hold_pct: float = self.settings.max_hold_pct

        # Risk management
        self.flat_stake_pct: float = self.settings.flat_stake_pct
        self.max_kelly_stake_pct: float = self.settings.max_kelly_stake_pct
        self.max_game_exposure_pct: float = self.settings.max_game_exposure_pct
        self.max_correlated_positions: int = self.settings.max_correlated_positions

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_opportunity(
        self,
        simulation_result: SimulationResult,
        odds_prop: dict,
        game_context: dict,
        feature_snapshot: dict,
    ) -> dict:
        """Evaluate a single player-game opportunity and return a decision dict.

        Args:
            simulation_result: Output of ``MonteCarloSimulator.simulate``.
            odds_prop: Dict with at least ``over_price``, ``under_price``,
                ``sportsbook``, ``snapshot_timestamp_utc``, ``line``,
                ``nba_game_id``, ``nba_player_id``.
            game_context: Dict with game-level context such as ``spread``,
                ``total``, ``tipoff_time_utc``, ``home_team_abbr``,
                ``away_team_abbr``.
            feature_snapshot: Dict with feature-level context including
                ``feature_snapshot_id``, ``tracking_available``,
                ``fallback_confidence``, ``player_archetype``,
                ``team_change_days``, ``model_run_id``.

        Returns:
            Decision dict matching the ``bet_decisions`` table schema.
        """
        now_utc = datetime.now(timezone.utc)

        # --- 1. Raw implied probabilities from book prices ----------------
        over_price = float(odds_prop["over_price"])
        under_price = float(odds_prop["under_price"])

        over_decimal = self._american_to_decimal(over_price)
        under_decimal = self._american_to_decimal(under_price)

        over_implied_raw = 1.0 / over_decimal if over_decimal > 0 else 0.5
        under_implied_raw = 1.0 / under_decimal if under_decimal > 0 else 0.5

        # --- 2. Remove vig using power method -----------------------------
        over_novig, under_novig = self._remove_vig_power(
            over_implied_raw, under_implied_raw
        )
        hold_pct = (over_implied_raw + under_implied_raw) - 1.0

        # --- 3. Model fair probabilities ----------------------------------
        model_p_over = simulation_result.p_over
        model_p_under = simulation_result.p_under

        # --- 4. Edge = model_prob - market_novig_prob ---------------------
        edge_over = model_p_over - over_novig
        edge_under = model_p_under - under_novig

        # --- 5. Expected value --------------------------------------------
        ev_over = (model_p_over * (over_decimal - 1.0)) - (1.0 - model_p_over)
        ev_under = (model_p_under * (under_decimal - 1.0)) - (1.0 - model_p_under)

        # --- 6. Recommended side ------------------------------------------
        if ev_over > ev_under and ev_over > 0:
            recommended_side = "over"
            best_ev = ev_over
            best_edge = edge_over
            best_model_prob = model_p_over
            best_decimal = over_decimal
        elif ev_under > 0:
            recommended_side = "under"
            best_ev = ev_under
            best_edge = edge_under
            best_model_prob = model_p_under
            best_decimal = under_decimal
        else:
            recommended_side = "no_bet"
            best_ev = max(ev_over, ev_under)
            best_edge = max(edge_over, edge_under)
            best_model_prob = model_p_over if ev_over >= ev_under else model_p_under
            best_decimal = over_decimal if ev_over >= ev_under else under_decimal

        # --- 7. Exclusion rules -------------------------------------------
        should_exclude, exclude_reason = self._check_exclusion_rules(
            feature_snapshot, game_context
        )

        # --- 8. Staleness check -------------------------------------------
        odds_timestamp = odds_prop.get("snapshot_timestamp_utc")
        if isinstance(odds_timestamp, str):
            odds_timestamp = datetime.fromisoformat(odds_timestamp)
        stale_check = False
        if odds_timestamp is not None:
            age_minutes = (now_utc - odds_timestamp.replace(tzinfo=timezone.utc)).total_seconds() / 60.0
            stale_check = age_minutes > self.max_stale_odds_minutes

        # --- 9. Trigger rules ---------------------------------------------
        should_bet, trigger_reason = self._check_trigger_rules(
            best_ev, best_edge, hold_pct, stale_check
        )

        # --- 10. Final decision -------------------------------------------
        if should_exclude:
            recommended_side = "no_bet"
            stake_pct = 0.0
            logger.info(
                "Excluded: player=%s game=%s reason=%s",
                odds_prop.get("nba_player_id"),
                odds_prop.get("nba_game_id"),
                exclude_reason,
            )
        elif not should_bet:
            recommended_side = "no_bet"
            stake_pct = 0.0
            logger.info(
                "No bet: player=%s game=%s reason=%s",
                odds_prop.get("nba_player_id"),
                odds_prop.get("nba_game_id"),
                trigger_reason,
            )
        else:
            stake_pct = self._compute_stake(
                best_model_prob, best_decimal, method="flat"
            )

        # --- Build decision dict ------------------------------------------
        decision = {
            "feature_snapshot_id": feature_snapshot.get("feature_snapshot_id"),
            "model_run_id": feature_snapshot.get("model_run_id"),
            "nba_game_id": odds_prop.get("nba_game_id"),
            "nba_player_id": odds_prop.get("nba_player_id"),
            "sportsbook": odds_prop.get("sportsbook"),
            "line": simulation_result.line,
            "odds_over": over_price,
            "odds_under": under_price,
            "model_p_over": round(model_p_over, 6),
            "model_p_under": round(model_p_under, 6),
            "fair_odds_over": round(simulation_result.fair_odds_over_american, 1),
            "fair_odds_under": round(simulation_result.fair_odds_under_american, 1),
            "edge_over": round(edge_over, 6),
            "edge_under": round(edge_under, 6),
            "recommended_side": recommended_side,
            "stake_pct": round(stake_pct, 6),
            "decision_timestamp_utc": now_utc,
            "tracking_available": feature_snapshot.get("tracking_available", False),
            # Fields populated after game settles
            "close_over_prob_novig": None,
            "close_under_prob_novig": None,
            "clv_prob_pts": None,
            "actual_3pm": None,
            "bet_result": None,
            "pnl_units": None,
            # Extra metadata (not stored in DB, useful for logging)
            "_ev_over": round(ev_over, 6),
            "_ev_under": round(ev_under, 6),
            "_hold_pct": round(hold_pct, 6),
            "_over_novig": round(over_novig, 6),
            "_under_novig": round(under_novig, 6),
            "_exclude_reason": exclude_reason if should_exclude else None,
            "_trigger_reason": trigger_reason if not should_bet else None,
            "_mean_3pm": simulation_result.mean_3pm,
            "_median_3pm": simulation_result.median_3pm,
            "_alt_line_probs": simulation_result.alt_line_probs,
        }

        return decision

    def line_shop(self, opportunities: list[dict]) -> dict:
        """Select the best available price from multiple books for the same player/game.

        Takes a list of decision dicts (from ``evaluate_opportunity``) for the
        same player and game across different sportsbooks.  Returns the single
        best opportunity for the recommended side.

        Args:
            opportunities: List of decision dicts from ``evaluate_opportunity``.

        Returns:
            The single best decision dict, or the first no-bet if none qualify.
        """
        if not opportunities:
            raise ValueError("No opportunities provided to line_shop")

        # Separate actionable bets from no-bets
        actionable = [o for o in opportunities if o["recommended_side"] != "no_bet"]

        if not actionable:
            logger.info("line_shop: no actionable opportunities, returning first no-bet")
            return opportunities[0]

        # Group by recommended side and pick the one with the best EV
        best: Optional[dict] = None
        best_ev: float = -float("inf")

        for opp in actionable:
            side = opp["recommended_side"]
            if side == "over":
                ev = opp["_ev_over"]
            else:
                ev = opp["_ev_under"]

            if ev > best_ev:
                best_ev = ev
                best = opp

        logger.info(
            "line_shop: selected %s @ %s (EV=%.4f) for player=%s game=%s",
            best["recommended_side"],
            best["sportsbook"],
            best_ev,
            best["nba_player_id"],
            best["nba_game_id"],
        )
        return best

    def check_correlation_limits(
        self,
        pending_decisions: list[dict],
        new_decision: dict,
    ) -> bool:
        """Check whether *new_decision* can be added without violating correlation limits.

        Rules enforced:
        1. Total game exposure must not exceed ``max_game_exposure_pct``.
        2. At most 1 position per player per 3PM side (no doubling up).
        3. At most ``max_correlated_positions`` same-game positions.
        4. No correlated stacking -- cannot take both over on player A and
           over on player B on the same team in the same game when both are
           high-volume shooters (prevents correlated blowups).

        Args:
            pending_decisions: Already-accepted decisions in the current slate.
            new_decision: The candidate decision to add.

        Returns:
            ``True`` if the new decision passes all correlation checks,
            ``False`` if it should be blocked.
        """
        if new_decision["recommended_side"] == "no_bet":
            return True

        new_game_id = new_decision["nba_game_id"]
        new_player_id = new_decision["nba_player_id"]
        new_side = new_decision["recommended_side"]
        new_stake = new_decision["stake_pct"]

        # --- Rule 1: Max game exposure ------------------------------------
        game_exposure = sum(
            d["stake_pct"]
            for d in pending_decisions
            if d["nba_game_id"] == new_game_id and d["recommended_side"] != "no_bet"
        )
        if game_exposure + new_stake > self.max_game_exposure_pct:
            logger.info(
                "Correlation limit: game exposure %.4f + %.4f > %.4f for game %s",
                game_exposure,
                new_stake,
                self.max_game_exposure_pct,
                new_game_id,
            )
            return False

        # --- Rule 2: Max 1 position per player per side -------------------
        for d in pending_decisions:
            if (
                d["nba_player_id"] == new_player_id
                and d["recommended_side"] == new_side
                and d["recommended_side"] != "no_bet"
            ):
                logger.info(
                    "Correlation limit: duplicate %s position on player %s",
                    new_side,
                    new_player_id,
                )
                return False

        # --- Rule 3: Max N correlated same-game positions -----------------
        same_game_count = sum(
            1
            for d in pending_decisions
            if d["nba_game_id"] == new_game_id and d["recommended_side"] != "no_bet"
        )
        if same_game_count >= self.max_correlated_positions:
            logger.info(
                "Correlation limit: already %d positions in game %s (max %d)",
                same_game_count,
                new_game_id,
                self.max_correlated_positions,
            )
            return False

        # --- Rule 4: Correlated stacking guard ----------------------------
        # Block adding a second "over" on a different player in the same
        # game if both are on the same team (same-team over stacking).
        if new_side == "over":
            new_team = new_decision.get("_team_abbr")
            for d in pending_decisions:
                if (
                    d["nba_game_id"] == new_game_id
                    and d["recommended_side"] == "over"
                    and d["nba_player_id"] != new_player_id
                    and d.get("_team_abbr") == new_team
                    and new_team is not None
                ):
                    logger.info(
                        "Correlation limit: same-team over stacking blocked "
                        "(player %s and %s on %s in game %s)",
                        new_player_id,
                        d["nba_player_id"],
                        new_team,
                        new_game_id,
                    )
                    return False

        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_exclusion_rules(
        self,
        feature_snapshot: dict,
        game_context: dict,
    ) -> tuple[bool, str]:
        """Apply exclusion rules that disqualify an opportunity entirely.

        Returns:
            ``(should_exclude, reason)`` -- if ``should_exclude`` is ``True``
            the opportunity must be skipped regardless of EV.
        """
        # 1. Tracking missing and fallback confidence too low
        tracking_available = feature_snapshot.get("tracking_available", False)
        fallback_confidence = feature_snapshot.get("fallback_confidence", 1.0)
        if not tracking_available and fallback_confidence < 0.5:
            return True, "tracking_missing_low_fallback_confidence"

        # 2. Player recently changed teams (< 10 games with new team)
        team_change_days = feature_snapshot.get("team_change_days")
        if team_change_days is not None and team_change_days < 30:
            return True, "recent_team_change"

        # 3. Projected minutes uncertainty too high (p90 - p10 > 20)
        minutes_p10 = feature_snapshot.get("minutes_p10", 0.0)
        minutes_p90 = feature_snapshot.get("minutes_p90", 48.0)
        if (minutes_p90 - minutes_p10) > 20.0:
            return True, "minutes_uncertainty_too_high"

        # 4. Blowout risk extreme + bench-volatile archetype
        spread = game_context.get("spread", 0.0)
        if spread is None:
            spread = 0.0
        archetype = feature_snapshot.get("player_archetype", "")
        if abs(spread) > 14.0 and archetype == "bench_microwave":
            return True, "blowout_risk_bench_volatile"

        # 5. Odds snapshot too stale (checked separately in evaluate but
        #    also caught here if explicit flag is set)
        odds_stale_flag = feature_snapshot.get("odds_stale", False)
        if odds_stale_flag:
            return True, "odds_snapshot_stale"

        # 6. Injury snapshot outdated
        injury_report_age_hours = feature_snapshot.get("injury_report_age_hours")
        if injury_report_age_hours is not None and injury_report_age_hours > 12.0:
            return True, "injury_snapshot_outdated"

        return False, ""

    def _check_trigger_rules(
        self,
        ev: float,
        edge: float,
        hold_pct: float,
        stale_check: bool,
    ) -> tuple[bool, str]:
        """Apply trigger rules that must all pass for a bet to fire.

        Returns:
            ``(should_bet, reason)`` -- if ``should_bet`` is ``False`` the
            reason explains which check failed.
        """
        if stale_check:
            return False, f"odds_stale (>{self.max_stale_odds_minutes}min)"

        if ev < self.min_ev_pct:
            return False, f"ev_too_low ({ev:.4f} < {self.min_ev_pct})"

        if edge < self.min_edge_prob_pts:
            return False, f"edge_too_small ({edge:.4f} < {self.min_edge_prob_pts})"

        if hold_pct > self.max_hold_pct:
            return False, f"hold_too_high ({hold_pct:.4f} > {self.max_hold_pct})"

        return True, "all_checks_passed"

    def _compute_stake(
        self,
        model_prob: float,
        decimal_odds: float,
        method: str = "flat",
    ) -> float:
        """Compute stake as a fraction of bankroll.

        Args:
            model_prob: Model's estimated win probability.
            decimal_odds: Decimal odds for the recommended side.
            method: ``"flat"`` or ``"quarter_kelly"``.

        Returns:
            Stake fraction (e.g. 0.005 = 0.5% of bankroll).
        """
        if method == "quarter_kelly":
            # Kelly criterion: f* = (bp - q) / b
            # where b = decimal_odds - 1, p = model_prob, q = 1 - p
            b = decimal_odds - 1.0
            if b <= 0:
                return 0.0
            q = 1.0 - model_prob
            kelly_full = (b * model_prob - q) / b
            if kelly_full <= 0:
                return 0.0
            quarter_kelly = 0.25 * kelly_full
            return min(quarter_kelly, self.max_kelly_stake_pct)
        else:
            # Flat staking
            return self.flat_stake_pct

    # ------------------------------------------------------------------
    # Odds conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _american_to_decimal(american: float) -> float:
        """Convert American odds to decimal odds.

        Examples:
            +150 -> 2.50
            -110 -> 1.909
        """
        if american > 0:
            return 1.0 + american / 100.0
        elif american < 0:
            return 1.0 + 100.0 / abs(american)
        else:
            return 1.0

    @staticmethod
    def _remove_vig_power(
        over_implied: float,
        under_implied: float,
    ) -> tuple[float, float]:
        """Remove vig from implied probabilities using the power method.

        The power method finds an exponent *k* such that:
            over_implied^k + under_implied^k = 1

        This distributes the overround proportionally to each side's
        probability, which more accurately reflects the true market
        when compared to the naive multiplicative method.

        Falls back to multiplicative devig if the power method does not
        converge (e.g., pathological inputs).

        Returns:
            ``(over_novig, under_novig)`` summing to 1.0.
        """
        # Guard against degenerate inputs
        over_implied = max(over_implied, 0.001)
        under_implied = max(under_implied, 0.001)
        total = over_implied + under_implied

        if abs(total - 1.0) < 1e-6:
            # Already fair
            return over_implied, under_implied

        # Power method: find k such that p_o^k + p_u^k = 1
        # Use bisection on k in (0, 10]
        lo, hi = 0.01, 10.0
        for _ in range(100):
            mid = (lo + hi) / 2.0
            val = over_implied ** mid + under_implied ** mid
            if val > 1.0:
                lo = mid
            else:
                hi = mid
            if abs(val - 1.0) < 1e-9:
                break

        k = (lo + hi) / 2.0
        over_fair = over_implied ** k
        under_fair = under_implied ** k

        # Normalize to ensure exact sum = 1
        fair_total = over_fair + under_fair
        if fair_total > 0:
            over_fair /= fair_total
            under_fair /= fair_total
        else:
            # Fallback to multiplicative
            over_fair = over_implied / total
            under_fair = under_implied / total

        return over_fair, under_fair
