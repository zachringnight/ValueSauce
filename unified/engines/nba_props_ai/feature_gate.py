from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional


class FeatureGateError(Exception):
    pass


def load_feature_gate(path: str) -> Dict[str, float]:
    p = Path(path).resolve()
    if not p.exists():
        raise FeatureGateError(f"Feature gate file not found: {p}")
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise FeatureGateError(f"Could not parse feature gate JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise FeatureGateError("Feature gate payload must be a JSON object.")
    weights = payload.get("feature_weights")
    if not isinstance(weights, dict):
        raise FeatureGateError("Feature gate JSON missing 'feature_weights' object.")
    out: Dict[str, float] = {}
    for key, raw in weights.items():
        try:
            out[str(key)] = float(raw)
        except Exception:
            continue
    return out


def apply_factor_gate(value: float, gate: Optional[Dict[str, float]], key: str) -> float:
    if gate is None:
        return float(value)
    if key not in gate:
        return float(value)
    weight = gate.get(key, 1.0)
    try:
        w = float(weight)
    except Exception:
        return float(value)
    if not (w == w):
        return float(value)
    w = min(max(w, 0.0), 1.0)
    return float(1.0 + (float(value) - 1.0) * w)

