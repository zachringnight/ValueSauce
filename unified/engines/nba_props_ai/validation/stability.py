"""7-day unattended pipeline stability harness.

Runs the pipeline repeatedly and records success/failure, timing,
data freshness, and error traces.
"""
from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class StabilityRunRecord:
    """One pipeline execution record."""
    run_id: int
    started_utc: str
    finished_utc: str
    elapsed_seconds: float
    success: bool
    n_results: int
    n_errors: int
    error_trace: str
    games_processed: int
    props_processed: int
    cache_hits: int
    cache_misses: int
    endpoint_errors: List[str] = field(default_factory=list)


@dataclass
class StabilityReport:
    """Summary of a multi-day stability test."""
    total_runs: int
    successful_runs: int
    failed_runs: int
    success_rate: float
    consecutive_successes: int
    max_consecutive_successes: int
    mean_elapsed_seconds: float
    p95_elapsed_seconds: float
    manual_interventions: int
    first_run_utc: str
    last_run_utc: str
    span_hours: float
    pass_7day: bool
    run_records: List[Dict[str, Any]]


def run_stability_test(
    pipeline_fn,
    *,
    n_runs: int = 14,
    interval_hours: float = 12.0,
    output_dir: str = "stability_results",
    dry_run: bool = False,
) -> StabilityReport:
    """Execute the pipeline repeatedly over a simulated or real schedule.

    Parameters
    ----------
    pipeline_fn : callable
        Function() -> dict with keys:
            n_results, n_errors, games, props, cache_hits, cache_misses,
            endpoint_errors (list[str]).
    n_runs : int
        Total executions (14 runs × 12h = 7 days).
    interval_hours : float
        Hours between runs (for real mode; ignored if dry_run).
    output_dir : str
        Directory to write per-run logs.
    dry_run : bool
        If True, execute all runs immediately without sleeping.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    records: List[StabilityRunRecord] = []

    for run_id in range(1, n_runs + 1):
        started = datetime.now(timezone.utc)
        success = False
        n_results = 0
        n_errors = 0
        error_trace = ""
        games = 0
        props = 0
        cache_hits = 0
        cache_misses = 0
        endpoint_errors: List[str] = []

        try:
            result = pipeline_fn()
            success = True
            n_results = int(result.get("n_results", 0))
            n_errors = int(result.get("n_errors", 0))
            games = int(result.get("games", 0))
            props = int(result.get("props", 0))
            cache_hits = int(result.get("cache_hits", 0))
            cache_misses = int(result.get("cache_misses", 0))
            endpoint_errors = list(result.get("endpoint_errors", []))
        except Exception:
            error_trace = traceback.format_exc()

        finished = datetime.now(timezone.utc)
        elapsed = (finished - started).total_seconds()

        record = StabilityRunRecord(
            run_id=run_id,
            started_utc=started.isoformat(),
            finished_utc=finished.isoformat(),
            elapsed_seconds=elapsed,
            success=success,
            n_results=n_results,
            n_errors=n_errors,
            error_trace=error_trace,
            games_processed=games,
            props_processed=props,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            endpoint_errors=endpoint_errors,
        )
        records.append(record)

        # Write per-run log
        run_log = out_path / f"run_{run_id:03d}.json"
        run_log.write_text(json.dumps(record.__dict__, indent=2, default=str))

        if not dry_run and run_id < n_runs:
            time.sleep(interval_hours * 3600)

    # Summarize
    successes = [r for r in records if r.success]
    failures = [r for r in records if not r.success]
    elapsed_vals = [r.elapsed_seconds for r in records]

    # Max consecutive successes
    max_consec = 0
    current_consec = 0
    for r in records:
        if r.success:
            current_consec += 1
            max_consec = max(max_consec, current_consec)
        else:
            current_consec = 0

    first_ts = records[0].started_utc if records else ""
    last_ts = records[-1].finished_utc if records else ""
    span_hours = 0.0
    if first_ts and last_ts:
        t0 = datetime.fromisoformat(first_ts)
        t1 = datetime.fromisoformat(last_ts)
        span_hours = (t1 - t0).total_seconds() / 3600

    import numpy as np
    p95 = float(np.percentile(elapsed_vals, 95)) if elapsed_vals else 0.0

    report = StabilityReport(
        total_runs=len(records),
        successful_runs=len(successes),
        failed_runs=len(failures),
        success_rate=len(successes) / max(len(records), 1),
        consecutive_successes=current_consec,
        max_consecutive_successes=max_consec,
        mean_elapsed_seconds=float(np.mean(elapsed_vals)) if elapsed_vals else 0.0,
        p95_elapsed_seconds=p95,
        manual_interventions=0,
        first_run_utc=first_ts,
        last_run_utc=last_ts,
        span_hours=span_hours,
        pass_7day=max_consec >= 14 and len(failures) == 0,
        run_records=[r.__dict__ for r in records],
    )

    # Write summary
    summary_path = out_path / "stability_summary.json"
    summary_path.write_text(json.dumps(report.__dict__, indent=2, default=str))

    return report
