from __future__ import annotations

import ast
import re
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .io_cards import CardError, load_game_card, load_props_card, validate_cards
from .pipeline import PipelineError, run_pipeline


class AppServiceError(Exception):
    """Structured error surface for UI callers."""

    def __init__(self, message: str, errors: Optional[List[Dict[str, Any]]] = None, cause: Optional[Exception] = None):
        super().__init__(message)
        self.errors = errors or []
        self.cause = cause


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sanitize_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.where(pd.notnull(out), "")
    return out


def _safe_literal_list(value: str) -> List[str]:
    try:
        parsed = ast.literal_eval(value)
    except Exception:
        return []
    if isinstance(parsed, (list, tuple)):
        return [str(x) for x in parsed]
    return []


def _translate_card_or_pipeline_error(exc: Exception) -> Dict[str, Any]:
    msg = str(exc)
    payload: Dict[str, Any] = {
        "type": "validation" if isinstance(exc, CardError) else "pipeline",
        "message": msg,
        "hint": "Review card inputs and retry." if isinstance(exc, CardError) else "Model pipeline failed; review game/prop context.",
    }

    m = re.search(r"^(Game|Props) card missing required columns:\s*(\[.*\])$", msg)
    if m:
        cols = _safe_literal_list(m.group(2))
        payload.update(
            {
                "field": "columns",
                "card": m.group(1).lower(),
                "missing_columns": cols,
                "hint": f"Add required columns to {m.group(1).lower()} card: {', '.join(cols)}",
            }
        )
        return payload

    m = re.search(r"^home must be 0 or 1 for game_key=(.+)$", msg)
    if m:
        payload.update(
            {
                "field": "home",
                "game_key": m.group(1),
                "hint": "Set 'home' to 1 for home team or 0 for away team.",
            }
        )
        return payload

    m = re.search(r"^Invalid market '(.+)' for (.+) (.+)$", msg)
    if m:
        payload.update(
            {
                "field": "market",
                "invalid_value": m.group(1),
                "player_name": m.group(2),
                "game_key": m.group(3),
                "hint": "Allowed markets: PTS, REB, AST, FG3M.",
            }
        )
        return payload

    m = re.search(r"^Invalid line for (.+) (.+) (.+): (.+)$", msg)
    if m:
        payload.update(
            {
                "field": "line",
                "player_name": m.group(1),
                "market": m.group(2),
                "game_key": m.group(3),
                "invalid_value": m.group(4),
                "hint": "Provide a numeric prop line, e.g. 24.5.",
            }
        )
        return payload

    m = re.search(r"^Invalid American odds(?: \(([^)]+)\))? for (.+) (.+) (.+): (.+)$", msg)
    if m:
        odds_field = m.group(1) or "odds"
        payload.update(
            {
                "field": odds_field,
                "player_name": m.group(2),
                "market": m.group(3),
                "game_key": m.group(4),
                "invalid_value": m.group(5),
                "hint": "Provide integer American odds, e.g. -110 or +120.",
            }
        )
        return payload

    m = re.search(r"^Prop references unknown game_key '(.+)'$", msg)
    if m:
        payload.update(
            {
                "field": "game_key",
                "game_key": m.group(1),
                "hint": "Every props card game_key must exist in the game card.",
            }
        )
        return payload

    m = re.search(r"^Duplicate game_key in game card:\s*(.+)$", msg)
    if m:
        payload.update(
            {
                "field": "game_key",
                "game_key": m.group(1),
                "hint": "Each game_key must appear only once in the game card.",
            }
        )
        return payload

    m = re.search(r"^(.+): players in both override_team_out and override_team_in:\s*(\[.*\])$", msg)
    if m:
        payload.update(
            {
                "field": "override_team_out/override_team_in",
                "game_key": m.group(1),
                "conflict_players": _safe_literal_list(m.group(2)),
                "hint": "A player cannot be marked both OUT and IN.",
            }
        )
        return payload

    m = re.search(r"^(.+): players in both override_opp_out and override_opp_in:\s*(\[.*\])$", msg)
    if m:
        payload.update(
            {
                "field": "override_opp_out/override_opp_in",
                "game_key": m.group(1),
                "conflict_players": _safe_literal_list(m.group(2)),
                "hint": "An opponent player cannot be marked both OUT and IN.",
            }
        )
        return payload

    m = re.search(r"^(.+): minutes override provided for OUT player '(.+)'$", msg)
    if m:
        payload.update(
            {
                "field": "minutes_caps/minutes_targets",
                "game_key": m.group(1),
                "player_name": m.group(2),
                "hint": "Remove minutes cap/target for players marked OUT.",
            }
        )
        return payload

    return payload


def _serialize_results(results: List[Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for item in results:
        row = asdict(item)
        dist = row.get("dist", {}) or {}
        row["asof_utc"] = item.asof_utc.isoformat()
        row["dist_name"] = dist.get("name")
        row["dist_params"] = dist.get("params", {})
        row["flags_str"] = ";".join(item.flags)
        row["drivers_str"] = " | ".join(item.drivers[:40])
        records.append(row)
    return records


def _serialize_games(games: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    payload: Dict[str, Dict[str, Any]] = {}
    for game_key, gs in games.items():
        row = asdict(gs)
        row["asof_utc"] = gs.asof_utc.isoformat()
        payload[game_key] = row
    return payload


def run_projection_job(
    game_card_df: pd.DataFrame,
    props_card_df: pd.DataFrame,
    season: str = "2025-26",
    asof: str = "now",
    refresh: str = "none",
    only_game_key: Optional[str] = None,
    cache_db: str = "nba_props_cache.sqlite",
) -> Dict[str, Any]:
    """
    Run the existing model pipeline from in-memory card DataFrames.

    Returns:
        {
          "results_df": pd.DataFrame,
          "results_records": List[dict],
          "games_state": Dict[str, dict],
          "injury_snapshot": Optional[dict],
          "run_meta": dict,
        }
    """
    started_at = _now_iso()
    if not isinstance(game_card_df, pd.DataFrame):
        raise AppServiceError("game_card_df must be a pandas DataFrame")
    if not isinstance(props_card_df, pd.DataFrame):
        raise AppServiceError("props_card_df must be a pandas DataFrame")

    game_card_df = _sanitize_frame(game_card_df)
    props_card_df = _sanitize_frame(props_card_df)

    try:
        with tempfile.TemporaryDirectory(prefix="nba_props_app_") as tmpdir:
            game_csv = Path(tmpdir) / "game_card.csv"
            props_csv = Path(tmpdir) / "props_card.csv"
            game_card_df.to_csv(game_csv, index=False, encoding="utf-8")
            props_card_df.to_csv(props_csv, index=False, encoding="utf-8")

            games = load_game_card(str(game_csv), asof)
            props = load_props_card(str(props_csv))
            validate_cards(games, props)
            results, injury_snapshot = run_pipeline(
                games=games,
                props=props,
                season=season,
                refresh=refresh,
                only_game_key=only_game_key,
                cache_db=cache_db,
            )
    except (CardError, PipelineError) as exc:
        structured = _translate_card_or_pipeline_error(exc)
        raise AppServiceError("Projection run failed due to card or pipeline validation.", errors=[structured], cause=exc) from exc
    except Exception as exc:
        structured = {
            "type": "unexpected",
            "message": str(exc),
            "hint": "Unexpected error during projection run. Verify environment dependencies and card data.",
        }
        raise AppServiceError("Projection run failed unexpectedly.", errors=[structured], cause=exc) from exc

    records = _serialize_results(results)
    results_df = pd.DataFrame(records)
    if not results_df.empty:
        results_df = results_df.sort_values(["game_key", "player_name", "market"]).reset_index(drop=True)

    run_meta = {
        "started_at_utc": started_at,
        "completed_at_utc": _now_iso(),
        "season": season,
        "asof": asof,
        "refresh": refresh,
        "only_game_key": only_game_key,
        "cache_db": cache_db,
        "input_game_rows": int(len(game_card_df)),
        "input_prop_rows": int(len(props_card_df)),
        "output_projection_rows": int(len(results_df)),
    }

    return {
        "results_df": results_df,
        "results_records": records,
        "games_state": _serialize_games(games),
        "injury_snapshot": injury_snapshot,
        "run_meta": run_meta,
    }


def rank_prop_plays(results_df: pd.DataFrame) -> pd.DataFrame:
    """Sort projections by recommended side edge first, then modeled side probability."""
    if results_df is None or len(results_df) == 0:
        return pd.DataFrame(columns=[] if results_df is None else results_df.columns)

    ranked = results_df.copy()
    if "edge_cents_side" not in ranked.columns:
        ranked["edge_cents_side"] = pd.to_numeric(ranked.get("edge_cents_over"), errors="coerce")
    if "model_prob_side" not in ranked.columns:
        ranked["model_prob_side"] = pd.to_numeric(ranked.get("p_over"), errors="coerce")

    ranked["_edge_side_sort"] = pd.to_numeric(ranked["edge_cents_side"], errors="coerce").fillna(-1e9)
    ranked["_model_prob_side_sort"] = pd.to_numeric(ranked["model_prob_side"], errors="coerce").fillna(-1e9)
    tie_breakers = [col for col in ("game_key", "player_name", "market") if col in ranked.columns]
    ranked = ranked.sort_values(
        ["_edge_side_sort", "_model_prob_side_sort", *tie_breakers],
        ascending=[False, False, *([True] * len(tie_breakers))],
        kind="mergesort",
    )
    ranked = ranked.drop(columns=["_edge_side_sort", "_model_prob_side_sort"]).reset_index(drop=True)
    return ranked
