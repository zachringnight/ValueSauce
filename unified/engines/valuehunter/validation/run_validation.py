"""CLI entry point for running the release-candidate validation pack."""

import argparse
import logging
import sys
from datetime import date, timedelta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="NBA 3PM Props Engine v1.1 — RC Validation Pack",
    )
    parser.add_argument(
        "--start-date", type=str, required=True,
        help="Evaluation start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date", type=str, default=None,
        help="Evaluation end date (YYYY-MM-DD), defaults to today",
    )
    parser.add_argument(
        "--train-window", type=int, default=180,
        help="Training window in days (default: 180)",
    )
    parser.add_argument(
        "--retrain-every", type=int, default=30,
        help="Retrain every N days (default: 30)",
    )
    parser.add_argument(
        "--stability-days", type=int, default=7,
        help="Number of stability check days (default: 7)",
    )
    parser.add_argument(
        "--min-paper-bets", type=int, default=100,
        help="Minimum paper bets required (default: 100)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="validation_output",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--mc-draws", type=int, default=25000,
        help="Monte Carlo simulation draws (default: 25000)",
    )
    args = parser.parse_args()

    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date) if args.end_date else date.today()

    logger.info("=" * 60)
    logger.info("NBA 3PM Props Engine v1.1 — RC Validation Pack")
    logger.info("Evaluation window: %s to %s", start, end)
    logger.info("=" * 60)

    # Build components
    from nba_props.config import get_settings
    from nba_props.utils.db import get_connection, Repository
    from nba_props.features import (
        ArchetypeClassifier,
        FeatureSnapshotBuilder,
        MakeRateFeatureBuilder,
        MinutesFeatureBuilder,
        OpportunityFeatureBuilder,
    )
    from nba_props.models import (
        MinutesModel,
        ThreePAModel,
        MakeRateModel,
        RollingAverageBaseline,
        DirectThreePMBaseline,
        BookmakerBaseline,
    )
    from nba_props.pricing import MonteCarloSimulator
    from nba_props.pricing.decision_engine import DecisionEngine
    from nba_props.backtest.leakage import LeakageDetector
    from nba_props.monitoring.drift import DriftMonitor
    from nba_props.validation.runner import ValidationRunner

    settings = get_settings()
    conn = get_connection(settings.database_url)
    repo = Repository(conn)

    feature_builder = FeatureSnapshotBuilder(
        minutes_builder=MinutesFeatureBuilder(),
        opportunity_builder=OpportunityFeatureBuilder(),
        make_rate_builder=MakeRateFeatureBuilder(),
        archetype_classifier=ArchetypeClassifier(),
    )

    minutes_model = MinutesModel()
    three_pa_model = ThreePAModel()
    make_rate_model = MakeRateModel()
    simulator = MonteCarloSimulator(n_simulations=args.mc_draws)
    decision_engine = DecisionEngine(settings=settings)
    leakage_detector = LeakageDetector()
    drift_monitor = DriftMonitor()

    baselines = {
        "rolling_avg": RollingAverageBaseline(),
        "direct_3pm": DirectThreePMBaseline(),
        "bookmaker": BookmakerBaseline(),
    }

    runner = ValidationRunner(
        repository=repo,
        feature_builder=feature_builder,
        minutes_model=minutes_model,
        three_pa_model=three_pa_model,
        make_rate_model=make_rate_model,
        simulator=simulator,
        decision_engine=decision_engine,
        baselines=baselines,
        leakage_detector=leakage_detector,
        drift_monitor=drift_monitor,
        settings=settings,
    )

    report = runner.run(
        start_date=start,
        end_date=end,
        train_window_days=args.train_window,
        retrain_every_days=args.retrain_every,
        stability_days=args.stability_days,
        min_paper_bets=args.min_paper_bets,
    )

    # Print report
    text = ValidationRunner.format_full_report(report)
    print(text)

    # Save report
    path = ValidationRunner.save_report(report, args.output_dir)
    logger.info("Report saved to %s", path)

    # Exit code based on promotion verdict
    sys.exit(0 if report.promoted else 1)


if __name__ == "__main__":
    main()
