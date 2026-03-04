from __future__ import annotations

import math
from typing import Tuple

def normal_over_prob(mu: float, sigma: float, line: float) -> float:
    if sigma <= 1e-9:
        return 1.0 if mu > line else 0.0
    z = (line - mu) / sigma
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return 1.0 - cdf

def negbin_params_from_mean_var(mu: float, var: float) -> Tuple[float, float]:
    if mu <= 0:
        return 1.0, 0.5
    if var <= mu + 1e-8:
        var = mu * 1.15 + 0.25
    p = mu / var
    p = max(1e-6, min(1.0 - 1e-6, p))
    n = mu * p / (1.0 - p)
    return float(n), float(p)

def negbin_over_prob(mu: float, var: float, line: float) -> float:
    k = int(math.floor(line + 1e-9)) + 1
    try:
        from scipy.stats import nbinom
        n, p = negbin_params_from_mean_var(mu, var)
        return float(1.0 - nbinom.cdf(k - 1, n, p))
    except Exception:
        sigma = math.sqrt(max(var, 1e-8))
        return normal_over_prob(mu, sigma, line)
