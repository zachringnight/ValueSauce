"""CLI entry point for running the daily pipeline."""

import argparse
import logging
import sys
from datetime import date, datetime

from nba_props.config import get_settings
from nba_props.utils.db import get_connection, Repository, run_all_migrations
from nba_props.ingestion.nba_stats import NBAStatsClient
from nba_props.ingestion.injury_reports import InjuryReportIngester
from nba_props.ingestion.odds_api import OddsAPIClient
from nba_props.features import (
    ArchetypeClassifier,
    FeatureSnapshotBuilder,
    MakeRateFeatureBuilder,
    MinutesFeatureBuilder,
    OpportunityFeatureBuilder,
)
from nba_props.models import MinutesModel, ThreePAModel, MakeRateModel
from nba_props.pricing import MonteCarloSimulator
from nba_props.pricing.decision_engine import DecisionEngine
from nba_props.jobs.daily_pipeline import DailyPipeline
from nba_props.jobs.ingestion_job import IngestionJob

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="NBA 3PM Props Engine - Daily Pipeline")
    parser.add_argument("--date", type=str, default=None, help="Target date (YYYY-MM-DD), defaults to today")
    parser.add_argument("--season", type=str, default="2025-26", help="NBA season (e.g. 2025-26)")
    parser.add_argument("--mode", choices=["ingest", "predict", "full"], default="full", help="Pipeline mode")
    parser.add_argument("--migrate", action="store_true", help="Run database migrations first")
    args = parser.parse_args()

    target_date = date.fromisoformat(args.date) if args.date else date.today()
    settings = get_settings()

    # Database setup
    conn = get_connection(settings.database_url)
    if args.migrate:
        run_all_migrations(conn, "sql/migrations")
        logger.info("Migrations complete")

    repo = Repository(conn)

    if args.mode in ("ingest", "full"):
        logger.info("Running ingestion for %s", target_date)
        nba_client = NBAStatsClient()
        injury_ingester = InjuryReportIngester()
        odds_client = OddsAPIClient(api_key=settings.odds_api_key) if settings.odds_api_key else None

        ingestion = IngestionJob(
            repository=repo,
            nba_client=nba_client,
            injury_ingester=injury_ingester,
            odds_client=odds_client,
        )
        results = ingestion.run_full_ingestion(target_date, args.season)
        logger.info("Ingestion results: %s", results)

    if args.mode in ("predict", "full"):
        logger.info("Running predictions for %s", target_date)

        # Build components
        feature_builder = FeatureSnapshotBuilder(
            minutes_builder=MinutesFeatureBuilder(),
            opportunity_builder=OpportunityFeatureBuilder(),
            make_rate_builder=MakeRateFeatureBuilder(),
            archetype_classifier=ArchetypeClassifier(),
        )

        minutes_model = MinutesModel()
        three_pa_model = ThreePAModel()
        make_rate_model = MakeRateModel()
        simulator = MonteCarloSimulator(n_simulations=settings.monte_carlo_draws)
        decision_engine = DecisionEngine(settings=settings)

        pipeline = DailyPipeline(
            repository=repo,
            feature_builder=feature_builder,
            minutes_model=minutes_model,
            three_pa_model=three_pa_model,
            make_rate_model=make_rate_model,
            simulator=simulator,
            decision_engine=decision_engine,
            settings=settings,
        )

        decisions = pipeline.run(target_date)
        bets = [d for d in decisions if d and d.get("recommended_side") != "no_bet"]
        logger.info("Generated %d decisions, %d bets", len(decisions), len(bets))


if __name__ == "__main__":
    main()
