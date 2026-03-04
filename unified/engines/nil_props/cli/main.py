"""CLI entry point for NBA Props Engine."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="nba-props", help="NBA Pregame Player-Assist Props Engine")
console = Console()


def _setup_logging(level: str = "INFO"):
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "nba_props.log"),
        ],
    )


def _get_deps():
    """Lazy-load dependencies to keep CLI startup fast."""
    from engines.nil_props.config.settings import get_settings
    from engines.nil_props.db import get_engine, get_session, init_db
    settings = get_settings()
    engine = get_engine()
    session = get_session(engine)
    return settings, engine, session


@app.command("init-db")
def init_db_cmd():
    """Initialize database schema."""
    _setup_logging()
    from engines.nil_props.db import get_engine, init_db
    engine = get_engine()
    init_db(engine)
    console.print("[green]Database initialized successfully.[/green]")


@app.command("ingest-historical")
def ingest_historical(
    season: str = typer.Option("2024-25", help="NBA season to ingest"),
):
    """Ingest historical data from all adapters."""
    _setup_logging()
    settings, engine, session = _get_deps()
    from engines.nil_props.db import init_db
    init_db(engine)

    from engines.nil_props.adapters.ingest import IngestPipeline
    from engines.nil_props.adapters.registry import get_all_adapters

    adapters = get_all_adapters(settings)
    pipeline = IngestPipeline(session, settings)

    total = 0
    for name, adapter in adapters.items():
        console.print(f"Ingesting from [cyan]{name}[/cyan] ({adapter.SOURCE_NAME})...")
        try:
            count = pipeline.run_historical(adapter, season)
            console.print(f"  Stored [green]{count}[/green] records")
            total += count
        except Exception as e:
            console.print(f"  [red]Error: {e}[/red]")

    console.print(f"\n[green]Total records ingested: {total}[/green]")


@app.command("reconcile")
def reconcile():
    """Run data quality checks and reconciliation."""
    _setup_logging()
    settings, engine, session = _get_deps()

    from engines.nil_props.reconciliation.reconciler import Reconciler

    reconciler = Reconciler(session)
    report = reconciler.run_full_check()

    if report.is_clean:
        console.print("[green]Data quality check passed — no issues found.[/green]")
    else:
        console.print("[yellow]Data quality issues found:[/yellow]")
        console.print(report.summary())
        for w in report.warnings[:10]:
            console.print(f"  [yellow]⚠ {w}[/yellow]")


@app.command("build-features")
def build_features(
    as_of_date: str = typer.Option(None, help="Build features up to this date (YYYY-MM-DD)"),
    output: str = typer.Option("data/processed/features.parquet", help="Output path"),
):
    """Build feature tables for all model layers."""
    _setup_logging()
    settings, engine, session = _get_deps()

    from engines.nil_props.features.builder import FeatureBuilder

    builder = FeatureBuilder(session)
    df = builder.build_all_features(as_of_date)

    if df.empty:
        console.print("[red]No features generated — is data ingested?[/red]")
        raise typer.Exit(1)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    console.print(f"[green]Built {len(df)} feature rows → {out_path}[/green]")
    console.print(f"  Columns: {len(df.columns)}")
    console.print(f"  Players: {df['player_id'].nunique()}")
    console.print(f"  Games: {df['game_id'].nunique()}")


@app.command("train-models")
def train_models(
    features_path: str = typer.Option("data/processed/features.parquet"),
    train_end: str = typer.Option(None, help="Train on data up to this date"),
):
    """Train minutes, opportunity, and conversion models."""
    _setup_logging()
    import pandas as pd
    from engines.nil_props.models.trainer import ModelTrainer

    df = pd.read_parquet(features_path)
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"])

    trainer = ModelTrainer()

    console.print("\n[cyan]Training minutes model...[/cyan]")
    min_result = trainer.train_minutes_model(df, train_end)
    _print_metrics("Minutes", min_result.metrics)

    console.print("\n[cyan]Training opportunity model...[/cyan]")
    opp_result = trainer.train_opportunity_model(df, train_end)
    _print_metrics("Opportunity", opp_result.metrics)

    console.print("\n[cyan]Training conversion model...[/cyan]")
    conv_result = trainer.train_conversion_model(df, train_end)
    _print_metrics("Conversion", conv_result.metrics)

    console.print("\n[cyan]Baseline (rolling 10-game avg)...[/cyan]")
    baseline = trainer.train_baseline_rolling(df)
    _print_metrics("Baseline", baseline)

    console.print("\n[green]All models trained successfully.[/green]")


@app.command("run-backtest")
def run_backtest(
    window: str = typer.Option("expanding", help="Window type: expanding or rolling"),
    min_edge: float = typer.Option(0.035, help="Minimum edge threshold"),
    books: str = typer.Option("draftkings,fanduel", help="Comma-separated sportsbook IDs (or 'all')"),
    output_dir: str = typer.Option("data/backtests", help="Output directory"),
):
    """Run time-based backtest with paper-bet logging."""
    _setup_logging()
    settings, engine, session = _get_deps()

    from engines.nil_props.backtest.engine import BacktestConfig, BacktestEngine

    sportsbooks = None if books == "all" else [b.strip() for b in books.split(",")]
    config = BacktestConfig(
        window_type=window,
        min_edge=min_edge,
        sportsbooks=sportsbooks,
    )

    bt = BacktestEngine(session, settings)
    result = bt.run(config)

    # Save results
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = result.summary_df()
    if not summary.empty:
        summary.to_csv(out_dir / f"{result.run_id}_bets.csv", index=False)
        console.print(f"\n[green]Backtest {result.run_id}[/green]")
        console.print(f"  Total bets: {len(result.paper_bets)}")
        console.print(f"  Hit rate: {result.hit_rate()}")
        console.print(f"  ROI: {result.roi()}")
        console.print(f"  Avg CLV: {result.avg_clv()}")

        # Segment breakdown
        for seg, metrics in result.segment_metrics.items():
            console.print(f"\n  [{seg}]")
            for k, v in metrics.items():
                console.print(f"    {k}: {v}")
    else:
        console.print("[yellow]No bets generated in backtest.[/yellow]")

    # Save metrics
    with open(out_dir / f"{result.run_id}_metrics.json", "w") as f:
        json.dump(result.segment_metrics, f, indent=2, default=str)


@app.command("run-daily-slate")
def run_daily_slate(
    date: str = typer.Option(None, help="Slate date (YYYY-MM-DD), default today"),
    min_edge: float = typer.Option(0.035, help="Minimum edge threshold"),
):
    """Generate recommendations for today's slate."""
    _setup_logging()
    settings, engine, session = _get_deps()

    from engines.nil_props.features.builder import FeatureBuilder
    from engines.nil_props.models.trainer import ModelTrainer
    from engines.nil_props.simulation.pricer import MonteCarloPricer
    from engines.nil_props.utils.odds import american_to_implied, compute_edge
    import numpy as np
    import pandas as pd

    target_date = date or datetime.utcnow().strftime("%Y-%m-%d")

    # Build features
    builder = FeatureBuilder(session)
    df = builder.build_all_features(target_date)

    if df.empty:
        console.print("[red]No data available for daily slate.[/red]")
        raise typer.Exit(1)

    # Train on all available data
    trainer = ModelTrainer()
    try:
        min_model = trainer.train_minutes_model(df, target_date)
        opp_model = trainer.train_opportunity_model(df, target_date)
        conv_model = trainer.train_conversion_model(df, target_date)
    except ValueError as e:
        console.print(f"[red]Training failed: {e}[/red]")
        raise typer.Exit(1)

    # Get scheduled games for target date
    from sqlalchemy import text as sql_text
    games = session.execute(
        sql_text("""SELECT g.game_id, g.home_team_id, g.away_team_id, g.game_date
                    FROM games g
                    WHERE g.game_date = :d AND g.status = 'scheduled'"""),
        {"d": target_date},
    ).fetchall()

    if not games:
        console.print(f"[yellow]No scheduled games for {target_date}.[/yellow]")
        # Show latest available data instead
        console.print("\n[cyan]Showing projections for latest completed games instead:[/cyan]")
        latest = df.tail(20)
        _show_projections(latest, min_model, opp_model, conv_model, settings, session)
        return

    console.print(f"\n[green]Daily Slate: {target_date}[/green]")
    console.print(f"Games: {len(games)}")


@app.command("generate-review")
def generate_review(
    backtest_dir: str = typer.Option("data/backtests", help="Directory with backtest results"),
    output: str = typer.Option("data/reports/review.md", help="Output report path"),
):
    """Generate a markdown review report from backtest results."""
    _setup_logging()
    import glob
    import pandas as pd

    bt_dir = Path(backtest_dir)
    csv_files = list(bt_dir.glob("*_bets.csv"))

    if not csv_files:
        console.print("[yellow]No backtest results found.[/yellow]")
        raise typer.Exit(1)

    all_bets = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    report_lines = [
        "# NBA Props Engine — Review Report",
        f"\nGenerated: {datetime.utcnow().isoformat()}",
        f"\n## Summary",
        f"\n- Total bets: {len(all_bets)}",
    ]

    resolved = all_bets[all_bets["result"].isin(["win", "loss"])]
    if not resolved.empty:
        hit = (resolved["result"] == "win").mean()
        report_lines.append(f"- Hit rate: {hit:.1%}")
        report_lines.append(f"- Avg edge: {resolved['edge'].mean():.3f}")

        if "clv" in resolved.columns:
            clv_vals = resolved["clv"].dropna()
            if not clv_vals.empty:
                report_lines.append(f"- Avg CLV: {clv_vals.mean():.4f}")

        # Biggest misses
        report_lines.append("\n## Biggest Misses")
        losses = resolved[resolved["result"] == "loss"].nlargest(5, "edge")
        for _, row in losses.iterrows():
            report_lines.append(
                f"- {row.get('player_name', row['player_id'])}: "
                f"{row['side']} {row['line']} (edge={row['edge']:.3f}, "
                f"actual={row.get('actual_assists', '?')})"
            )

        # By side
        report_lines.append("\n## By Side")
        for side in ["over", "under"]:
            subset = resolved[resolved["side"] == side]
            if not subset.empty:
                hr = (subset["result"] == "win").mean()
                report_lines.append(f"- {side}: {len(subset)} bets, {hr:.1%} hit rate")

    report_text = "\n".join(report_lines)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_text)
    console.print(f"[green]Review report saved to {out_path}[/green]")


def _print_metrics(name: str, metrics: dict):
    table = Table(title=name)
    table.add_column("Metric")
    table.add_column("Value")
    for k, v in metrics.items():
        if isinstance(v, float):
            table.add_row(k, f"{v:.4f}")
        else:
            table.add_row(k, str(v))
    console.print(table)


def _show_projections(df, min_model, opp_model, conv_model, settings, session):
    """Show projections for a set of rows."""
    import numpy as np
    from engines.nil_props.simulation.pricer import MonteCarloPricer

    pricer = MonteCarloPricer(
        n_draws=settings.monte_carlo_draws,
        seed=settings.monte_carlo_seed,
    )

    table = Table(title="Projections")
    table.add_column("Player")
    table.add_column("Proj Min")
    table.add_column("Proj Ast")
    table.add_column("Actual Ast")

    for _, row in df.iterrows():
        try:
            min_pred = float(np.clip(min_model.predict(row.to_frame().T)[0], 0, 48))
            opp_pred = float(max(0, opp_model.predict(row.to_frame().T)[0]))
            conv_pred = float(np.clip(conv_model.predict(row.to_frame().T)[0], 0, 1))

            opp_rate = opp_pred / max(min_pred, 1)
            pricing = pricer.price(
                player_id=row["player_id"],
                game_id=row["game_id"],
                line=5.5,
                proj_minutes_mean=min_pred,
                proj_minutes_std=None,
                proj_opportunity_rate=opp_rate,
                proj_opportunity_std=None,
                proj_conversion_rate=conv_pred,
                proj_conversion_std=None,
            )

            table.add_row(
                str(row.get("full_name", row["player_id"])),
                f"{min_pred:.1f}",
                f"{pricing.proj_assists:.1f}",
                str(row.get("assists", "?")),
            )
        except Exception:
            pass

    console.print(table)


@app.command("export-paper-bets")
def export_paper_bets(
    backtest_dir: str = typer.Option("data/backtests"),
    output: str = typer.Option("data/reports/paper_bets.csv"),
):
    """Export all paper bets to CSV."""
    import pandas as pd

    bt_dir = Path(backtest_dir)
    csv_files = list(bt_dir.glob("*_bets.csv"))

    if not csv_files:
        console.print("[yellow]No backtest results found.[/yellow]")
        raise typer.Exit(1)

    all_bets = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_bets.to_csv(out_path, index=False)
    console.print(f"[green]Exported {len(all_bets)} paper bets to {out_path}[/green]")


if __name__ == "__main__":
    app()
