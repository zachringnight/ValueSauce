from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import List, Optional

def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)

def parse_asof(asof: str) -> datetime:
    if str(asof).lower() == "now":
        return utcnow()
    s = str(asof).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(microsecond=0)

def norm_name(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip())

def name_key(name: str) -> str:
    """Canonical key for fuzzy player-name matching (case/punctuation insensitive)."""
    return re.sub(r"[^a-z0-9]+", "", norm_name(name).lower())

def split_semicolon_list(value: str) -> List[str]:
    if value is None:
        return []
    items = [norm_name(x) for x in str(value).split(";")]
    return [x for x in items if x]

def parse_minutes_value(v) -> float:
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s:
        return 0.0
    if ":" in s:
        try:
            mm, ss = s.split(":")
            return float(mm) + float(ss)/60.0
        except Exception:
            return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0

def american_to_implied_prob(odds: int) -> float:
    if odds == 0:
        raise ValueError("Odds cannot be 0")
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return (-odds) / ((-odds) + 100.0)

def implied_prob_to_american(p: float) -> int:
    if not (0.0 < p < 1.0):
        raise ValueError("Probability must be in (0,1)")
    dec = 1.0 / p
    if dec >= 2.0:
        return int(round((dec - 1.0) * 100.0))
    return int(round(-100.0 / (dec - 1.0)))

def safe_clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def eb_shrink_rate(obs_rate: float, obs_n: float, prior_rate: float, prior_n: float) -> float:
    if obs_n < 0:
        obs_n = 0
    if prior_n <= 0:
        return obs_rate
    return (obs_rate * obs_n + prior_rate * prior_n) / (obs_n + prior_n)

def now_iso() -> str:
    return utcnow().isoformat()
