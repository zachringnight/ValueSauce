from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .utils import american_to_implied_prob

REQUIRED_BET_LOG_COLUMNS = [
    "asof_utc",
    "game_key",
    "player",
    "market",
    "line",
    "odds",
    "side",
    "result",
]

WIN_RESULTS = {"WIN", "WON", "W", "1"}
LOSS_RESULTS = {"LOSS", "LOST", "L", "0"}
PUSH_RESULTS = {"PUSH", "VOID", "CANCEL", "REFUND", "P"}
PENDING_RESULTS = {"PENDING", "OPEN", "UNGRADED"}
OVER_SIDES = {"OVER", "O"}
UNDER_SIDES = {"UNDER", "U"}


@dataclass(frozen=True)
class ProfitSummary:
    total_rows: int
    graded_bets: int
    wins: int
    losses: int
    pushes_or_voids: int
    total_units: float
    roi: float
    win_rate: float
    avg_edge: Optional[float]
    expected_units: Optional[float]
    pending: int = 0


def _normalize_result(raw: Any) -> str:
    token = str(raw or "").strip().upper()
    if token in WIN_RESULTS:
        return "WIN"
    if token in LOSS_RESULTS:
        return "LOSS"
    if token in PUSH_RESULTS:
        return "PUSH"
    if token in PENDING_RESULTS:
        return "PENDING"
    raise ValueError(f"Unsupported result value '{raw}'. Use WIN/LOSS/PUSH/PENDING.")


def _normalize_side(raw: Any) -> str:
    token = str(raw or "").strip().upper()
    if token in OVER_SIDES:
        return "OVER"
    if token in UNDER_SIDES:
        return "UNDER"
    raise ValueError(f"Unsupported side value '{raw}'. Use OVER/UNDER.")


def payout_ratio_from_american(odds: int) -> float:
    if odds == 0:
        raise ValueError("American odds cannot be 0.")
    if odds > 0:
        return float(odds) / 100.0
    return 100.0 / abs(float(odds))


def settled_units(odds: int, normalized_result: str) -> float:
    if normalized_result == "WIN":
        return payout_ratio_from_american(odds)
    if normalized_result == "LOSS":
        return -1.0
    if normalized_result == "PUSH":
        return 0.0
    if normalized_result == "PENDING":
        return 0.0
    raise ValueError(f"Unexpected normalized result '{normalized_result}'.")


def _choose_model_side_probability(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)

    # Check both column names: p_model_side (backtest format) and
    # model_prob_side (bet log / results CSV format).
    for col in ("p_model_side", "model_prob_side"):
        if col in df.columns:
            direct = pd.to_numeric(df[col], errors="coerce")
            out = out.fillna(direct)

    if "p_over" in df.columns:
        p_over = pd.to_numeric(df["p_over"], errors="coerce")
        side_from_over = np.where(df["side_norm"] == "OVER", p_over, 1.0 - p_over)
        out = out.fillna(pd.Series(side_from_over, index=df.index, dtype=float))

    return out


def prepare_bet_log(raw_df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_BET_LOG_COLUMNS if c not in raw_df.columns]
    if missing:
        raise ValueError(f"Bet log missing required columns: {missing}")

    df = raw_df.copy()
    df["asof_utc"] = pd.to_datetime(df["asof_utc"], errors="coerce", utc=True)
    if df["asof_utc"].isna().any():
        raise ValueError("Bet log has invalid asof_utc values.")

    df["player"] = df["player"].astype(str).str.strip()
    df["market"] = df["market"].astype(str).str.strip().str.upper()
    df["line"] = pd.to_numeric(df["line"], errors="coerce")
    if df["line"].isna().any():
        raise ValueError("Bet log has non-numeric line values.")

    df["odds"] = pd.to_numeric(df["odds"], errors="coerce")
    if df["odds"].isna().any():
        raise ValueError("Bet log has non-numeric odds values.")
    if (df["odds"] == 0).any():
        raise ValueError("Bet log has odds=0, which is invalid for American odds.")
    df["odds"] = df["odds"].astype(int)

    df["side_norm"] = df["side"].apply(_normalize_side)
    df["result_norm"] = df["result"].apply(_normalize_result)

    df["is_graded"] = df["result_norm"].isin(["WIN", "LOSS"])
    df["is_win"] = df["result_norm"].eq("WIN")
    df["units"] = [settled_units(int(o), r) for o, r in zip(df["odds"], df["result_norm"])]
    df["implied_prob"] = df["odds"].apply(lambda x: american_to_implied_prob(int(x)))
    df["payout_ratio"] = df["odds"].apply(lambda x: payout_ratio_from_american(int(x)))

    model_prob = _choose_model_side_probability(df)
    model_prob = pd.to_numeric(model_prob, errors="coerce").clip(1e-6, 1.0 - 1e-6)
    df["model_prob_side"] = model_prob
    df["edge_vs_market"] = df["model_prob_side"] - df["implied_prob"]
    df["expected_units"] = (df["model_prob_side"] * df["payout_ratio"]) - (1.0 - df["model_prob_side"])

    return df


def summarize_profitability(prepared_df: pd.DataFrame) -> ProfitSummary:
    graded = prepared_df[prepared_df["is_graded"]]
    graded_count = int(len(graded))
    wins = int(graded["is_win"].sum())
    losses = int(graded_count - wins)
    pushes = int((prepared_df["result_norm"] == "PUSH").sum())
    pending = int((prepared_df["result_norm"] == "PENDING").sum())
    units = float(prepared_df["units"].sum())
    roi = float(units / graded_count) if graded_count else 0.0
    win_rate = float(wins / graded_count) if graded_count else 0.0

    edge_series = pd.to_numeric(graded["edge_vs_market"], errors="coerce").dropna()
    avg_edge = float(edge_series.mean()) if len(edge_series) else None

    ev_series = pd.to_numeric(graded["expected_units"], errors="coerce").dropna()
    expected_units = float(ev_series.sum()) if len(ev_series) else None

    return ProfitSummary(
        total_rows=int(len(prepared_df)),
        graded_bets=graded_count,
        wins=wins,
        losses=losses,
        pushes_or_voids=pushes,
        total_units=units,
        roi=roi,
        win_rate=win_rate,
        avg_edge=avg_edge,
        expected_units=expected_units,
        pending=pending,
    )


def summarize_by_market(prepared_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for market, chunk in prepared_df.groupby("market", dropna=False):
        graded = chunk[chunk["is_graded"]]
        graded_count = int(len(graded))
        wins = int(graded["is_win"].sum())
        units = float(chunk["units"].sum())
        roi = float(units / graded_count) if graded_count else 0.0
        win_rate = float(wins / graded_count) if graded_count else 0.0

        edge_series = pd.to_numeric(graded["edge_vs_market"], errors="coerce").dropna()
        ev_series = pd.to_numeric(graded["expected_units"], errors="coerce").dropna()

        rows.append(
            {
                "market": str(market),
                "total_rows": int(len(chunk)),
                "graded_bets": graded_count,
                "wins": wins,
                "losses": int(graded_count - wins),
                "pushes_or_voids": int((chunk["result_norm"] == "PUSH").sum()),
                "pending": int((chunk["result_norm"] == "PENDING").sum()),
                "units": units,
                "roi": roi,
                "win_rate": win_rate,
                "avg_edge": float(edge_series.mean()) if len(edge_series) else np.nan,
                "expected_units": float(ev_series.sum()) if len(ev_series) else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    return out.sort_values(["units", "roi"], ascending=[False, False]).reset_index(drop=True)


def calibration_metrics(prepared_df: pd.DataFrame, bins: int = 10) -> Optional[Dict[str, Any]]:
    return calibration_metrics_for_prob(prepared_df, prob_col="model_prob_side", bins=bins)


def calibration_metrics_for_prob(prepared_df: pd.DataFrame, prob_col: str, bins: int = 10) -> Optional[Dict[str, Any]]:
    graded = prepared_df[prepared_df["is_graded"]].copy()
    if len(graded) == 0:
        return None

    if prob_col not in graded.columns:
        return None
    p = pd.to_numeric(graded[prob_col], errors="coerce").dropna()
    if len(p) == 0:
        return None

    graded = graded.loc[p.index].copy()
    y = graded["is_win"].astype(float)
    p = p.clip(1e-6, 1.0 - 1e-6)

    brier = float(np.mean((p - y) ** 2))
    log_loss = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    bucket = np.floor(p * bins).astype(int).clip(0, bins - 1)
    bin_rows: List[Dict[str, Any]] = []
    for b in range(bins):
        mask = bucket == b
        if int(mask.sum()) == 0:
            continue
        bin_p = p[mask]
        bin_y = y[mask]
        bin_rows.append(
            {
                "bin": int(b),
                "count": int(mask.sum()),
                "p_mean": float(bin_p.mean()),
                "win_rate": float(bin_y.mean()),
                "calibration_gap": float(bin_y.mean() - bin_p.mean()),
            }
        )

    return {
        "n": int(len(p)),
        "brier": brier,
        "log_loss": log_loss,
        "reliability_bins": bin_rows,
    }


def _sigmoid(x):
    x = np.clip(x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x))


def _fit_isotonic_pav(p: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(p) == 0:
        return {"kind": "constant", "value": 0.5}
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    y = np.clip(y, 0.0, 1.0)

    order = np.argsort(p)
    x = p[order]
    z = y[order]

    means: List[float] = []
    weights: List[float] = []
    x_max: List[float] = []

    for xi, yi in zip(x, z):
        means.append(float(yi))
        weights.append(1.0)
        x_max.append(float(xi))
        while len(means) >= 2 and means[-2] > means[-1]:
            m1, w1 = means[-2], weights[-2]
            m2, w2 = means[-1], weights[-1]
            new_w = w1 + w2
            means[-2] = (m1 * w1 + m2 * w2) / new_w
            weights[-2] = new_w
            x_max[-2] = x_max[-1]
            means.pop()
            weights.pop()
            x_max.pop()

    return {"kind": "isotonic", "x_max": np.array(x_max, dtype=float), "y_hat": np.array(means, dtype=float)}


def _fit_platt(p: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(p) == 0:
        return {"kind": "constant", "value": 0.5}
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    y = np.clip(y, 0.0, 1.0)
    if np.all(y == y[0]):
        return {"kind": "constant", "value": float(y[0])}

    x = np.log(p / (1.0 - p))
    reg = 1e-4

    def loss(theta):
        a, b = float(theta[0]), float(theta[1])
        pred = _sigmoid(a * x + b)
        eps = 1e-9
        nll = -np.mean(y * np.log(pred + eps) + (1.0 - y) * np.log(1.0 - pred + eps))
        return float(nll + reg * (a * a + b * b))

    res = minimize(loss, x0=np.array([1.0, 0.0], dtype=float), method="L-BFGS-B")
    if not res.success:
        return {"kind": "constant", "value": float(np.mean(y))}
    a, b = float(res.x[0]), float(res.x[1])
    return {"kind": "platt", "a": a, "b": b}


def _fit_calibrator(method: str, p: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    if len(p) == 0:
        return {"kind": "constant", "value": 0.5}
    if method == "isotonic":
        return _fit_isotonic_pav(p, y)
    if method == "platt":
        return _fit_platt(p, y)
    raise ValueError(f"Unsupported calibration method '{method}'")


def _apply_calibrator(model: Dict[str, Any], p: float) -> float:
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    kind = model.get("kind")
    if kind == "constant":
        return float(np.clip(model.get("value", 0.5), 1e-6, 1.0 - 1e-6))
    if kind == "platt":
        a = float(model.get("a", 1.0))
        b = float(model.get("b", 0.0))
        x = np.log(p / (1.0 - p))
        return float(np.clip(_sigmoid(a * x + b), 1e-6, 1.0 - 1e-6))
    if kind == "isotonic":
        x_max = np.asarray(model.get("x_max", []), dtype=float)
        y_hat = np.asarray(model.get("y_hat", []), dtype=float)
        if len(x_max) == 0 or len(y_hat) == 0:
            return p
        idx = int(np.searchsorted(x_max, p, side="left"))
        idx = min(max(idx, 0), len(y_hat) - 1)
        return float(np.clip(y_hat[idx], 1e-6, 1.0 - 1e-6))
    return p


def _derive_time_bucket(series: pd.Series, bucket_minutes: int) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    if vals.notna().any():
        width = max(int(bucket_minutes), 1)
        lo = (np.floor(vals / width) * width).astype("Int64")
        hi = (lo + width).astype("Int64")
        return lo.astype(str) + "-" + hi.astype(str) + "m"
    return series.astype(str).replace({"": "__NA__", "nan": "__NA__", "None": "__NA__"})


def apply_walkforward_calibration(
    prepared_df: pd.DataFrame,
    *,
    method: str = "isotonic",
    min_train_rows: int = 80,
    group_cols: Optional[List[str]] = None,
    time_bucket_col: Optional[str] = None,
    time_bucket_minutes: int = 60,
) -> pd.DataFrame:
    """
    Calibrate model probabilities with strict walk-forward fitting.
    Each row is calibrated using only prior graded rows in the same group.
    """
    method = str(method).strip().lower()
    if method not in {"isotonic", "platt", "none"}:
        raise ValueError("method must be one of: isotonic, platt, none")

    out = prepared_df.copy()
    out["model_prob_side_raw"] = pd.to_numeric(out["model_prob_side"], errors="coerce").clip(1e-6, 1.0 - 1e-6)
    out["model_prob_side_calibrated"] = out["model_prob_side_raw"]
    out["calibration_train_rows"] = 0
    out["calibration_group"] = ""

    if method == "none":
        out["p_over_calibrated"] = np.where(
            out["side_norm"] == "OVER",
            out["model_prob_side_calibrated"],
            1.0 - out["model_prob_side_calibrated"],
        )
        return out

    group_cols = list(group_cols or ["market"])
    for col in group_cols:
        if col not in out.columns:
            raise ValueError(f"group column '{col}' not found in prepared bet log")
    if time_bucket_col:
        if time_bucket_col not in out.columns:
            raise ValueError(f"time bucket column '{time_bucket_col}' not found in prepared bet log")
        out["_time_bucket"] = _derive_time_bucket(out[time_bucket_col], bucket_minutes=time_bucket_minutes)
        group_cols.append("_time_bucket")

    work = out.reset_index().rename(columns={"index": "_orig_idx"})
    work = work.sort_values(["asof_utc", "_orig_idx"]).reset_index(drop=True)

    states: Dict[Any, Dict[str, Any]] = {}
    for _, row in work.iterrows():
        gvals = [row[c] for c in group_cols]
        key = tuple("__NA__" if pd.isna(v) else str(v) for v in gvals)
        state = states.setdefault(key, {"p": [], "y": [], "model": None, "dirty": True})

        if state["dirty"]:
            if len(state["p"]) >= int(min_train_rows):
                state["model"] = _fit_calibrator(method, np.array(state["p"], dtype=float), np.array(state["y"], dtype=float))
            else:
                state["model"] = None
            state["dirty"] = False

        idx = int(row["_orig_idx"])
        p_raw = row["model_prob_side_raw"]
        p_cal = p_raw
        if pd.notna(p_raw) and state["model"] is not None:
            p_cal = _apply_calibrator(state["model"], float(p_raw))

        out.at[idx, "model_prob_side_calibrated"] = p_cal
        out.at[idx, "calibration_train_rows"] = int(len(state["p"]))
        out.at[idx, "calibration_group"] = "|".join(key)

        if bool(row.get("is_graded")) and pd.notna(p_raw):
            state["p"].append(float(p_raw))
            state["y"].append(1.0 if bool(row.get("is_win")) else 0.0)
            state["dirty"] = True

    out["model_prob_side_calibrated"] = pd.to_numeric(out["model_prob_side_calibrated"], errors="coerce").clip(1e-6, 1.0 - 1e-6)
    out["p_over_calibrated"] = np.where(
        out["side_norm"] == "OVER",
        out["model_prob_side_calibrated"],
        1.0 - out["model_prob_side_calibrated"],
    )
    out["edge_vs_market_calibrated"] = out["model_prob_side_calibrated"] - out["implied_prob"]
    out["expected_units_calibrated"] = (out["model_prob_side_calibrated"] * out["payout_ratio"]) - (1.0 - out["model_prob_side_calibrated"])
    return out
