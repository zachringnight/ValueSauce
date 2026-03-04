"""Backtest engine — time-based evaluation with CLV tracking."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from engines.nil_props.config.settings import Settings
from engines.nil_props.features.builder import FeatureBuilder
from engines.nil_props.models.trainer import ModelTrainer
from engines.nil_props.simulation.pricer import MonteCarloPricer, PricingResult
from engines.nil_props.utils.odds import american_to_implied, compute_edge

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""

    window_type: str = "expanding"  # 'expanding' or 'rolling'
    rolling_window_days: int = 90
    min_train_games: int = 50
    min_edge: float = 0.035
    min_projected_minutes: float = 24.0
    min_prior_games: int = 2  # require 2+ prior games before betting (increase to 5+ with real data)
    max_bets_per_player_game: int = 1  # deduplicate across sportsbooks
    sportsbooks: list[str] | None = None  # None = all; e.g. ["draftkings", "fanduel"]
    n_draws: int = 10_000
    seed: int = 42


@dataclass
class PaperBet:
    """A single paper-trade bet."""

    bet_timestamp: datetime
    player_id: str
    game_id: str
    player_name: str
    side: str  # 'over' or 'under'
    line: float
    market_price: float  # American odds
    fair_price: float  # American odds
    fair_prob: float
    market_prob: float
    edge: float
    proj_minutes: float
    proj_assists: float
    actual_assists: int | None = None
    actual_minutes: float | None = None
    closing_line: float | None = None
    closing_price: float | None = None
    result: str | None = None  # 'win', 'loss', 'push'
    clv: float | None = None


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    run_id: str
    config: BacktestConfig
    paper_bets: list[PaperBet] = field(default_factory=list)
    model_metrics: dict = field(default_factory=dict)
    segment_metrics: dict = field(default_factory=dict)

    def summary_df(self) -> pd.DataFrame:
        if not self.paper_bets:
            return pd.DataFrame()
        rows = []
        for b in self.paper_bets:
            rows.append({
                "player_id": b.player_id,
                "game_id": b.game_id,
                "player_name": b.player_name,
                "side": b.side,
                "line": b.line,
                "market_price": b.market_price,
                "fair_price": b.fair_price,
                "edge": b.edge,
                "proj_minutes": b.proj_minutes,
                "proj_assists": b.proj_assists,
                "actual_assists": b.actual_assists,
                "actual_minutes": b.actual_minutes,
                "result": b.result,
                "clv": b.clv,
            })
        return pd.DataFrame(rows)

    def hit_rate(self) -> float | None:
        resolved = [b for b in self.paper_bets if b.result in ("win", "loss")]
        if not resolved:
            return None
        return sum(1 for b in resolved if b.result == "win") / len(resolved)

    def roi(self) -> float | None:
        resolved = [b for b in self.paper_bets if b.result in ("win", "loss")]
        if not resolved:
            return None
        total_wagered = len(resolved)
        total_return = sum(
            self._bet_payout(b.market_price) if b.result == "win" else -1.0
            for b in resolved
        )
        return total_return / total_wagered

    def avg_clv(self) -> float | None:
        clvs = [b.clv for b in self.paper_bets if b.clv is not None]
        if not clvs:
            return None
        return float(np.mean(clvs))

    @staticmethod
    def _bet_payout(american_price: float) -> float:
        """Return net profit on a 1-unit bet at American odds."""
        if american_price >= 100:
            return american_price / 100.0
        elif american_price <= -100:
            return 100.0 / abs(american_price)
        return 0.0


class BacktestEngine:
    """Runs time-based backtests with paper-bet logging."""

    def __init__(self, session: Session, settings: Settings):
        self.session = session
        self.settings = settings

    def run(self, config: BacktestConfig | None = None) -> BacktestResult:
        """Run a full backtest."""
        config = config or BacktestConfig()
        run_id = f"bt_{uuid.uuid4().hex[:8]}"
        result = BacktestResult(run_id=run_id, config=config)

        # Build full feature table
        builder = FeatureBuilder(self.session)
        full_df = builder.build_all_features()

        if full_df.empty:
            logger.warning("No data for backtest")
            return result

        # Get unique game dates for time splits
        game_dates = sorted(full_df["game_date"].unique())
        if len(game_dates) < 5:
            logger.warning("Too few game dates for backtest")
            return result

        # Determine test windows
        # Use last 20% of dates as test period
        split_idx = int(len(game_dates) * 0.6)
        test_dates = game_dates[split_idx:]

        trainer = ModelTrainer()
        pricer = MonteCarloPricer(n_draws=config.n_draws, seed=config.seed)

        # Load odds for edge comparison (filtered by sportsbook if configured)
        odds_df = self._load_odds(config.sportsbooks)

        for test_date in test_dates:
            train_end = test_date - pd.Timedelta(days=1)

            # Filter training data
            if config.window_type == "rolling":
                train_start = train_end - pd.Timedelta(days=config.rolling_window_days)
                train_df = full_df[
                    (full_df["game_date"] >= train_start) & (full_df["game_date"] <= train_end)
                ]
            else:
                train_df = full_df[full_df["game_date"] <= train_end]

            if len(train_df) < config.min_train_games:
                continue

            test_df = full_df[full_df["game_date"] == test_date]
            if test_df.empty:
                continue

            try:
                # Train models on historical data only
                min_model = trainer.train_minutes_model(train_df, str(train_end))
                opp_model = trainer.train_opportunity_model(train_df, str(train_end))
                conv_model = trainer.train_conversion_model(train_df, str(train_end))
                direct_model = trainer.train_direct_assist_model(train_df, str(train_end))
            except ValueError as e:
                logger.warning(f"Training failed for {test_date}: {e}")
                continue

            # Generate predictions for test games
            seen_player_game = set()  # deduplicate bets
            for _, row in test_df.iterrows():
                pid = row["player_id"]
                gid = row["game_id"]

                # Skip players with insufficient prior games
                prior_count = len(
                    train_df[train_df["player_id"] == pid]
                )
                if prior_count < config.min_prior_games:
                    continue

                try:
                    bets = self._evaluate_player(
                        row, min_model, opp_model, conv_model,
                        pricer, odds_df, config,
                        direct_model=direct_model,
                    )
                    # Enforce max bets per player-game
                    for bet in bets:
                        key = (bet.player_id, bet.game_id, bet.side)
                        if key not in seen_player_game:
                            seen_player_game.add(key)
                            result.paper_bets.append(bet)
                except Exception as e:
                    logger.debug(f"Eval failed for {pid}: {e}")

        # Resolve bets
        self._resolve_bets(result, full_df, odds_df)

        # Compute segment metrics
        self._compute_segments(result)

        logger.info(
            f"Backtest {run_id}: {len(result.paper_bets)} bets, "
            f"hit_rate={result.hit_rate()}, roi={result.roi()}, avg_clv={result.avg_clv()}"
        )

        return result

    def _evaluate_player(
        self,
        row: pd.Series,
        min_model,
        opp_model,
        conv_model,
        pricer: MonteCarloPricer,
        odds_df: pd.DataFrame,
        config: BacktestConfig,
        direct_model=None,
    ) -> list[PaperBet]:
        """Evaluate one player for potential bets."""
        bets = []
        pid = row["player_id"]
        gid = row["game_id"]

        # Build prediction input: extract numeric features as a dict, then DataFrame
        # Use model's training medians for NaN imputation (consistent with training)
        def _make_input(model_result) -> pd.DataFrame:
            vals = {}
            medians = model_result.feature_medians or {}
            for f in model_result.features:
                v = row.get(f)
                try:
                    vals[f] = [float(v)] if v is not None and pd.notna(v) else [medians.get(f, 0.0)]
                except (TypeError, ValueError):
                    vals[f] = [medians.get(f, 0.0)]
            return pd.DataFrame(vals)

        # Predict minutes
        min_pred = float(min_model.predict(_make_input(min_model))[0])
        min_pred = np.clip(min_pred, 0, 48)

        if min_pred < config.min_projected_minutes:
            return bets

        # Three-layer prediction: minutes × opportunity_rate × conversion
        opp_pred = float(opp_model.predict(_make_input(opp_model))[0])
        opp_pred = max(0, opp_pred)

        conv_pred = float(conv_model.predict(_make_input(conv_model))[0])
        conv_pred = np.clip(conv_pred, 0, 1)

        # Direct assist prediction — anchors against multiplicative regression error
        if direct_model is not None:
            direct_pred = float(direct_model.predict(_make_input(direct_model))[0])
            direct_pred = max(0, direct_pred)

            # Three-layer expected assists
            three_layer_ast = opp_pred * conv_pred

            # Blend: 50% direct, 50% three-layer
            # This prevents the three-layer multiplicative regression from dominating
            blended_ast = 0.5 * direct_pred + 0.5 * three_layer_ast

            # Adjust opp_pred and conv_pred to reproduce blended target
            # Keep conv_pred as is, scale opp_pred so opp_pred * conv_pred = blended_ast
            if conv_pred > 0:
                opp_pred = blended_ast / conv_pred
            else:
                opp_pred = blended_ast

        # Get non-closing odds for this player/game
        player_odds = odds_df[
            (odds_df["player_id"] == pid)
            & (odds_df["game_id"] == gid)
            & (odds_df["is_closing"] == 0)
        ]
        if player_odds.empty:
            return bets
        # Deduplicate: keep one line per sportsbook
        player_odds = player_odds.drop_duplicates(
            subset=["sportsbook_id", "line"], keep="last"
        )

        for _, odds_row in player_odds.iterrows():
            line = odds_row["line"]
            over_price = odds_row["over_price"]
            under_price = odds_row["under_price"]

            # Opportunity rate = potential_assists / minute
            opp_rate = opp_pred / max(min_pred, 1)

            # Compute per-minute std from model residuals
            opp_residual_std = opp_model.residual_std or 3.0
            opp_rate_std = opp_residual_std / max(min_pred, 1)

            pricing = pricer.price(
                player_id=pid,
                game_id=gid,
                line=line,
                proj_minutes_mean=min_pred,
                proj_minutes_std=min_model.residual_std,
                proj_opportunity_rate=opp_rate,
                proj_opportunity_std=opp_rate_std,
                proj_conversion_rate=conv_pred,
                proj_conversion_std=conv_model.residual_std,
            )

            # Check over side
            try:
                market_over_prob = american_to_implied(over_price)
            except ValueError:
                continue

            over_edge = compute_edge(pricing.fair_over_prob, market_over_prob)
            if over_edge >= config.min_edge:
                bets.append(PaperBet(
                    bet_timestamp=datetime.utcnow(),
                    player_id=pid,
                    game_id=gid,
                    player_name=row.get("full_name", pid),
                    side="over",
                    line=line,
                    market_price=over_price,
                    fair_price=pricing.fair_over_price,
                    fair_prob=pricing.fair_over_prob,
                    market_prob=market_over_prob,
                    edge=over_edge,
                    proj_minutes=min_pred,
                    proj_assists=pricing.proj_assists,
                ))

            # Check under side
            try:
                market_under_prob = american_to_implied(under_price)
            except ValueError:
                continue

            under_edge = compute_edge(pricing.fair_under_prob, market_under_prob)
            if under_edge >= config.min_edge:
                bets.append(PaperBet(
                    bet_timestamp=datetime.utcnow(),
                    player_id=pid,
                    game_id=gid,
                    player_name=row.get("full_name", pid),
                    side="under",
                    line=line,
                    market_price=under_price,
                    fair_price=pricing.fair_under_price,
                    fair_prob=pricing.fair_under_prob,
                    market_prob=market_under_prob,
                    edge=under_edge,
                    proj_minutes=min_pred,
                    proj_assists=pricing.proj_assists,
                ))

        return bets

    def _resolve_bets(
        self, result: BacktestResult, full_df: pd.DataFrame, odds_df: pd.DataFrame
    ):
        """Fill in actual results and CLV for paper bets."""
        for bet in result.paper_bets:
            # Get actual stats
            actual = full_df[
                (full_df["player_id"] == bet.player_id)
                & (full_df["game_id"] == bet.game_id)
            ]
            if not actual.empty:
                row = actual.iloc[0]
                bet.actual_assists = int(row["assists"]) if pd.notna(row["assists"]) else None
                bet.actual_minutes = float(row["minutes"]) if pd.notna(row["minutes"]) else None

                if bet.actual_assists is not None:
                    if bet.side == "over":
                        if bet.actual_assists > bet.line:
                            bet.result = "win"
                        elif bet.actual_assists < bet.line:
                            bet.result = "loss"
                        else:
                            bet.result = "push"
                    else:
                        if bet.actual_assists < bet.line:
                            bet.result = "win"
                        elif bet.actual_assists > bet.line:
                            bet.result = "loss"
                        else:
                            bet.result = "push"

            # CLV: compare bet-time prob vs closing prob
            closing = odds_df[
                (odds_df["player_id"] == bet.player_id)
                & (odds_df["game_id"] == bet.game_id)
                & (odds_df["is_closing"] == True)
            ]
            if not closing.empty:
                close_row = closing.iloc[0]
                try:
                    if bet.side == "over":
                        close_prob = american_to_implied(close_row["over_price"])
                    else:
                        close_prob = american_to_implied(close_row["under_price"])
                    bet.closing_price = (
                        close_row["over_price"] if bet.side == "over"
                        else close_row["under_price"]
                    )
                    bet.closing_line = close_row["line"]
                    bet.clv = bet.fair_prob - close_prob
                except (ValueError, KeyError):
                    pass

    def _compute_segments(self, result: BacktestResult):
        """Compute metrics by segment."""
        if not result.paper_bets:
            return

        df = result.summary_df()
        resolved = df[df["result"].isin(["win", "loss"])]
        if resolved.empty:
            return

        # Overall
        result.segment_metrics["overall"] = {
            "total_bets": len(resolved),
            "hit_rate": result.hit_rate(),
            "roi": result.roi(),
            "avg_clv": result.avg_clv(),
            "avg_edge": float(resolved["edge"].mean()),
        }

        # By side
        for side in ["over", "under"]:
            subset = resolved[resolved["side"] == side]
            if not subset.empty:
                wins = (subset["result"] == "win").sum()
                result.segment_metrics[f"side_{side}"] = {
                    "total_bets": len(subset),
                    "hit_rate": wins / len(subset),
                    "avg_edge": float(subset["edge"].mean()),
                }

    def _load_odds(self, sportsbooks: list[str] | None = None) -> pd.DataFrame:
        """Load odds snapshots, optionally filtered to specific sportsbooks."""
        rows = self.session.execute(
            text("""SELECT o.player_id, o.game_id, o.sportsbook_id,
                           o.line, o.over_price, o.under_price,
                           o.snapshot_timestamp, 0 as is_closing
                    FROM odds_props_snapshots o
                    UNION ALL
                    SELECT c.player_id, c.game_id, c.sportsbook_id,
                           c.line, c.over_price, c.under_price,
                           NULL as snapshot_timestamp, 1 as is_closing
                    FROM odds_props_closing c""")
        ).fetchall()
        if not rows:
            return pd.DataFrame(columns=[
                "player_id", "game_id", "sportsbook_id", "line",
                "over_price", "under_price", "snapshot_timestamp", "is_closing",
            ])
        df = pd.DataFrame([dict(r._mapping) for r in rows])
        if sportsbooks:
            df = df[df["sportsbook_id"].isin(sportsbooks)]
        return df
