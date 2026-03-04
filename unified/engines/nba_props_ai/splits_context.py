from __future__ import annotations

import json
import os
from io import StringIO
import pandas as pd
from typing import Dict

from .cache import SQLiteCache
from .utils import safe_clip

STAT_MAP = {
    "PTS": "PTS", "REB": "REB", "AST": "AST", "FG3M": "FG3M",
    "STL": "STL", "BLK": "BLK", "TOV": "TOV", "FGM": "FGM",
    "FGA": "FGA", "FTM": "FTM", "FTA": "FTA", "MIN": "MIN",
}

class PlayerSplitsContext:
    def __init__(self, player_id: int, season: str, cache: SQLiteCache):
        self.player_id = int(player_id)
        self.season = season
        self.cache = cache
        self._loaded: Dict[str, pd.DataFrame] | None = None

    @staticmethod
    def _empty_payload() -> Dict[str, pd.DataFrame]:
        return {
            "overall": pd.DataFrame(),
            "location": pd.DataFrame(),
            "win_loss": pd.DataFrame(),
            "month": pd.DataFrame(),
            "all_star": pd.DataFrame(),
            "starter": pd.DataFrame(),
            "rest": pd.DataFrame(),
        }

    def _load(self) -> Dict[str, pd.DataFrame]:
        if self._loaded is not None:
            return self._loaded
        params = {"player_id": self.player_id, "season": self.season}
        hit = self.cache.get("playerdashboardbygeneralsplits", params)
        if hit:
            try:
                payload = json.loads(hit.data_json)
                out: Dict[str, pd.DataFrame] = {}
                for k, v in payload.items():
                    text = v if isinstance(v, str) else json.dumps(v)
                    out[k] = pd.read_json(StringIO(text)) if text else pd.DataFrame()
                self._loaded = out
                return out
            except Exception:
                # Corrupted/partial cache entry; fall through to refetch path.
                pass
        if str(os.getenv("NBA_PROPS_CACHE_ONLY_ENDPOINTS", "")).strip().lower() in {"1", "true", "yes", "on"}:
            self._loaded = self._empty_payload()
            return self._loaded
        from nba_api.stats.endpoints import playerdashboardbygeneralsplits
        from .nba_data import _run_endpoint_call
        import time
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                time.sleep(0.6)
                resp = _run_endpoint_call(
                    lambda: playerdashboardbygeneralsplits.PlayerDashboardByGeneralSplits(
                        player_id=self.player_id,
                        season=self.season,
                        per_mode_detailed="PerGame",
                    ),
                    label="playerdashboardbygeneralsplits",
                )
                dfs = _run_endpoint_call(
                    lambda: resp.get_data_frames(),
                    label="playerdashboardbygeneralsplits:get_data_frames",
                )
                payload = self._empty_payload()
                ordered = ["overall", "location", "win_loss", "month", "all_star", "starter", "rest"]
                for idx, key in enumerate(ordered):
                    payload[key] = dfs[idx] if idx < len(dfs) else pd.DataFrame()
                self.cache.set("playerdashboardbygeneralsplits", params, {k: v.to_json() for k, v in payload.items()})
                self._loaded = payload
                return payload
            except Exception as exc:
                last_exc = exc
                time.sleep(0.75 + attempt * 0.5)

        payload = self._empty_payload()
        self.cache.set("playerdashboardbygeneralsplits", params, {k: v.to_json() for k, v in payload.items()})
        if last_exc is not None:
            print(
                "[splits_context] warning: PlayerDashboardByGeneralSplits failed "
                f"for player_id={self.player_id} season={self.season}: {last_exc}"
            )
        self._loaded = payload
        return payload

    def _overall(self, stat: str) -> float:
        splits = self._load()
        col = STAT_MAP.get(stat, stat)
        df = splits["overall"]
        if len(df) == 0 or col not in df.columns:
            return 0.0
        return float(df[col].iloc[0])

    def rest_factor(self, stat: str, rest_days: int) -> float:
        splits = self._load()
        col = STAT_MAP.get(stat, stat)
        overall = self._overall(stat)
        if overall <= 1e-9:
            return 1.0
        rest_df = splits["rest"]
        if len(rest_df) == 0 or col not in rest_df.columns:
            return 1.0
        labels = {0: "0 Days Rest", 1: "1 Days Rest", 2: "2 Days Rest", 3: "3 Days Rest", 4: "4 Days Rest", 5: "5 Days Rest"}
        label = labels.get(int(rest_days), "6+ Days Rest")
        row = rest_df[rest_df["GROUP_VALUE"] == label]
        if len(row) == 0:
            return 1.0
        gp = int(row["GP"].iloc[0]) if "GP" in row.columns else 0
        val = float(row[col].iloc[0])
        raw = val / overall
        weight = min(max(gp, 0) / 12.0, 1.0)
        return 1.0 + (raw - 1.0) * weight

    def home_away_factor(self, stat: str, is_home: bool) -> float:
        splits = self._load()
        col = STAT_MAP.get(stat, stat)
        overall = self._overall(stat)
        if overall <= 1e-9:
            return 1.0
        loc_df = splits["location"]
        if len(loc_df) == 0 or col not in loc_df.columns:
            return 1.0
        label = "Home" if is_home else "Road"
        row = loc_df[loc_df["GROUP_VALUE"] == label]
        if len(row) == 0:
            return 1.0
        gp = int(row["GP"].iloc[0]) if "GP" in row.columns else 0
        raw = float(row[col].iloc[0]) / overall
        weight = min(max(gp, 0) / 20.0, 1.0) * 0.85
        return 1.0 + (raw - 1.0) * weight

    def recent_trend_factor(self, stat: str) -> float:
        splits = self._load()
        col = STAT_MAP.get(stat, stat)
        overall = self._overall(stat)
        if overall <= 1e-9:
            return 1.0
        month_df = splits["month"]
        if len(month_df) == 0 or col not in month_df.columns:
            return 1.0
        month_order = ["October", "November", "December", "January", "February", "March", "April"]
        latest = None
        latest_gp = 0
        for m in reversed(month_order):
            row = month_df[month_df["GROUP_VALUE"] == m]
            if len(row):
                latest = float(row[col].iloc[0])
                latest_gp = int(row["GP"].iloc[0]) if "GP" in row.columns else 0
                break
        if latest is None or latest_gp < 3:
            return 1.0
        raw = latest / overall
        weight = min(latest_gp / 18.0, 0.35)
        return 1.0 + (raw - 1.0) * weight

    def composite_factor(self, stat: str, is_home: bool | None = None, rest_days: int | None = None) -> float:
        adjustments = []
        if rest_days is not None:
            adjustments.append(self.rest_factor(stat, int(rest_days)))
        if is_home is not None:
            adjustments.append(self.home_away_factor(stat, bool(is_home)))
        adjustments.append(self.recent_trend_factor(stat))
        total_dev = sum(a - 1.0 for a in adjustments)
        if abs(total_dev) > 0.10:
            sign = 1.0 if total_dev > 0 else -1.0
            total_dev = 0.10 * sign + (abs(total_dev) - 0.10) * 0.5 * sign
        return safe_clip(1.0 + total_dev, 0.85, 1.20)
