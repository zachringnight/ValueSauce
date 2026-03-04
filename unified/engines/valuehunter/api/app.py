"""FastAPI application for NBA 3PM Props Engine."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import date, datetime

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional

from ..config import get_settings

logger = logging.getLogger(__name__)


# --- Request / Response Models ---

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    timestamp: str


class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    train_start_date: Optional[str] = None
    train_end_date: Optional[str] = None
    metrics: Optional[dict] = None


class ModelsResponse(BaseModel):
    models: list[ModelInfo]


class DecisionResponse(BaseModel):
    game_id: str
    player_id: str
    market: str = "player_3pt_made"
    book: str
    line: float
    odds_over: float
    odds_under: float
    model_p_over: float
    model_p_under: float
    fair_odds_over: float
    fair_odds_under: float
    recommended_side: str
    edge: float
    stake_pct: float
    decision_timestamp: str
    model_version: str
    feature_snapshot_id: int
    tracking_available: bool
    injury_snapshot_timestamp: Optional[str] = None
    odds_snapshot_timestamp: Optional[str] = None


class DecisionsResponse(BaseModel):
    date: str
    decisions: list[DecisionResponse]
    total: int
    bets: int
    passes: int


class BacktestRequest(BaseModel):
    start_date: str
    end_date: str
    freeze_offset_minutes: int = Field(default=-60, description="Minutes before tipoff to freeze features")
    also_score_t30: bool = False


class BacktestResponse(BaseModel):
    run_id: str
    status: str
    start_date: str
    end_date: str
    minutes_mae: Optional[float] = None
    three_pa_mae: Optional[float] = None
    log_loss: Optional[float] = None
    brier_score: Optional[float] = None
    n_predictions: int = 0
    n_bets: int = 0
    roi: Optional[float] = None
    clv_mean: Optional[float] = None


class FeatureSnapshotResponse(BaseModel):
    feature_snapshot_id: int
    nba_game_id: str
    nba_player_id: str
    freeze_timestamp_utc: str
    tracking_available: bool
    archetype: Optional[str] = None
    minutes_p10: Optional[float] = None
    minutes_p50: Optional[float] = None
    minutes_p90: Optional[float] = None
    pred_3pa_mean: Optional[float] = None
    model_p_over: Optional[float] = None
    model_p_under: Optional[float] = None
    feature_json: Optional[dict] = None


class BetRecord(BaseModel):
    decision_id: int
    game_id: str
    player_id: str
    sportsbook: str
    line: float
    recommended_side: str
    edge: float
    stake_pct: float
    actual_3pm: Optional[int] = None
    bet_result: Optional[str] = None
    pnl_units: Optional[float] = None


class BetsResponse(BaseModel):
    date: str
    bets: list[BetRecord]
    total: int
    pending: int
    settled: int


# --- Application ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    logger.info("NBA 3PM Props Engine starting up")
    yield
    logger.info("NBA 3PM Props Engine shutting down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="NBA 3PM Props Engine",
        description="Pregame NBA player 3-point props pricing and decision system",
        version="1.1.0",
        lifespan=lifespan,
    )

    @app.get("/v1/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(
            status="ok",
            version="1.1.0",
            timestamp=datetime.utcnow().isoformat(),
        )

    @app.get("/v1/models", response_model=ModelsResponse)
    async def list_models():
        # In production, query model_runs table
        return ModelsResponse(models=[
            ModelInfo(
                model_name="minutes_model",
                model_version="1.0",
            ),
            ModelInfo(
                model_name="three_pa_model",
                model_version="1.0",
            ),
            ModelInfo(
                model_name="make_rate_model",
                model_version="1.0",
            ),
        ])

    @app.post("/v1/decisions", response_model=DecisionsResponse)
    async def generate_decisions(
        date: str = Query(..., description="Date in YYYY-MM-DD format"),
    ):
        """
        Generate bet/no-bet decisions for all eligible players on a given date.

        This is the main entry point for the engine. It:
        1. Fetches all games for the date
        2. Identifies eligible players
        3. Builds feature snapshots
        4. Runs the model pipeline (minutes -> 3PA -> make rate -> MC pricing)
        5. Compares to market odds
        6. Returns bet/no-bet decisions
        """
        # Placeholder: In production, this orchestrates the full pipeline
        return DecisionsResponse(
            date=date,
            decisions=[],
            total=0,
            bets=0,
            passes=0,
        )

    @app.post("/v1/backtest/run", response_model=BacktestResponse)
    async def run_backtest(request: BacktestRequest):
        """Run a research backtest over the specified date range."""
        import uuid
        run_id = str(uuid.uuid4())[:8]

        return BacktestResponse(
            run_id=run_id,
            status="queued",
            start_date=request.start_date,
            end_date=request.end_date,
        )

    @app.get("/v1/feature_snapshot/{snapshot_id}", response_model=FeatureSnapshotResponse)
    async def get_feature_snapshot(snapshot_id: int):
        """Retrieve a frozen feature snapshot by ID."""
        raise HTTPException(status_code=404, detail="Feature snapshot not found")

    @app.get("/v1/bets/{date}", response_model=BetsResponse)
    async def get_bets(date: str):
        """Get all bet decisions for a given date."""
        return BetsResponse(
            date=date,
            bets=[],
            total=0,
            pending=0,
            settled=0,
        )

    return app
