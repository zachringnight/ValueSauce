from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .models import ProjectionResult
from .utils import american_to_implied_prob, implied_prob_to_american


class CalibrationRuntimeError(Exception):
    pass


def _clip_prob(p: float) -> float:
    v = float(p)
    if not math.isfinite(v):
        return 0.5
    return float(np.clip(v, 1e-6, 1.0 - 1e-6))


def _safe_float(value: Any, *, default: float) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _as_sequence(value: Any) -> List[Any]:
    if isinstance(value, (str, bytes)):
        return []
    if isinstance(value, Iterable):
        return list(value)
    return []


def _coerce_isotonic_pairs(x_raw: Any, y_raw: Any) -> Tuple[np.ndarray, np.ndarray]:
    x_vals = _as_sequence(x_raw)
    y_vals = _as_sequence(y_raw)
    if not x_vals or not y_vals:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    xs: List[float] = []
    ys: List[float] = []
    for x_item, y_item in zip(x_vals, y_vals):
        x = _safe_float(x_item, default=float("nan"))
        y = _safe_float(y_item, default=float("nan"))
        if math.isfinite(x) and math.isfinite(y):
            xs.append(float(x))
            ys.append(float(y))

    if not xs:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _market_prob_and_edge(prob: float, odds_american: Optional[int]) -> Tuple[Optional[float], Optional[float]]:
    if odds_american is None:
        return None, None
    market_prob = american_to_implied_prob(int(odds_american))
    edge_cents = (float(prob) - float(market_prob)) * 100.0
    return float(market_prob), float(edge_cents)


def _pick_recommended_side(edge_over: Optional[float], edge_under: Optional[float]) -> Optional[str]:
    if edge_over is None and edge_under is None:
        return None
    if edge_over is None:
        return "under"
    if edge_under is None:
        return "over"
    return "over" if float(edge_over) >= float(edge_under) else "under"


def _sigmoid(x: float) -> float:
    x = max(min(float(x), 35.0), -35.0)
    return float(1.0 / (1.0 + math.exp(-x)))


def _apply_model(model: Dict[str, Any], p: float) -> float:
    p = _clip_prob(p)
    kind = str(model.get("kind") or model.get("method") or "").strip().lower()
    if kind == "platt":
        a = _safe_float(model.get("a", 1.0), default=1.0)
        b = _safe_float(model.get("b", 0.0), default=0.0)
        x = math.log(p / (1.0 - p))
        calibrated = _sigmoid(a * x + b)
        return _clip_prob(calibrated if math.isfinite(calibrated) else p)
    if kind == "isotonic":
        x, y = _coerce_isotonic_pairs(model.get("x_max", []), model.get("y_hat", []))
        if len(x) == 0 or len(y) == 0:
            return p
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        idx = int(np.searchsorted(x, p, side="left"))
        idx = min(max(idx, 0), len(y) - 1)
        calibrated = float(y[idx])
        return _clip_prob(calibrated if math.isfinite(calibrated) else p)
    if kind == "constant":
        calibrated = _safe_float(model.get("value", 0.5), default=p)
        return _clip_prob(calibrated)
    return p


def load_calibrator(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise CalibrationRuntimeError(f"Calibrator file not found: {path}")
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CalibrationRuntimeError(f"Could not parse calibrator JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise CalibrationRuntimeError("Calibrator payload must be a JSON object.")
    return payload


def _pick_market_model(calibrator: Dict[str, Any], market: str) -> Optional[Dict[str, Any]]:
    markets = calibrator.get("markets")
    if isinstance(markets, dict):
        exact = markets.get(str(market))
        if isinstance(exact, dict):
            return exact

        # Backward/hand-authored artifacts may vary key casing.
        market_upper = str(market).upper()
        for key, candidate in markets.items():
            if str(key).upper() == market_upper and isinstance(candidate, dict):
                return candidate

    model = calibrator.get("global")
    if isinstance(model, dict):
        return model
    return None


def apply_calibration_to_results(
    results: Iterable[ProjectionResult],
    calibrator: Dict[str, Any],
) -> List[ProjectionResult]:
    out: List[ProjectionResult] = []
    for r in results:
        model = _pick_market_model(calibrator, r.market)
        if not model:
            out.append(r)
            continue

        p_over_raw = _clip_prob(r.p_over)
        p_over = _apply_model(model, p_over_raw)
        p_under = _clip_prob(1.0 - p_over)

        odds_over = r.odds_over_american if r.odds_over_american is not None else r.odds_american
        odds_under = r.odds_under_american
        market_prob_over, edge_over = _market_prob_and_edge(p_over, odds_over)
        market_prob_under, edge_under = _market_prob_and_edge(p_under, odds_under)
        recommended_side = _pick_recommended_side(edge_over, edge_under)
        recommended_odds = odds_over if recommended_side == "over" else odds_under if recommended_side == "under" else None
        model_prob_side = p_over if recommended_side == "over" else p_under if recommended_side == "under" else None
        market_prob_side = market_prob_over if recommended_side == "over" else market_prob_under if recommended_side == "under" else None
        edge_side = edge_over if recommended_side == "over" else edge_under if recommended_side == "under" else None
        eligible = bool(recommended_side and (recommended_odds is not None))
        method = str(model.get("kind") or model.get("method") or "calibrator").lower()
        flags = list(r.flags)
        flags.append(f"calibrated:{method}")

        out.append(
            replace(
                r,
                p_over=float(p_over),
                p_under=float(p_under),
                fair_over_odds=implied_prob_to_american(p_over),
                fair_under_odds=implied_prob_to_american(p_under),
                market_prob_over=market_prob_over,
                market_prob_under=market_prob_under,
                edge_cents_over=edge_over,
                edge_cents_under=edge_under,
                recommended_side=recommended_side,
                recommended_odds_american=recommended_odds,
                model_prob_side=model_prob_side,
                market_prob_side=market_prob_side,
                edge_cents_side=edge_side,
                eligible_for_recommendation=eligible,
                flags=flags,
            )
        )
    return out
