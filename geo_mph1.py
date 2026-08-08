"""M/PH/1 batch-service queue using a Coxian-2 representation of service time.

This is a heuristic, not a numerically exact PH/PH/1 solution.  It mirrors
``geo_mg1k`` but reports the SCV-aware correction.  The exact PH/PH/1
solution is left to a future reference implementation.
"""
from __future__ import annotations

import math
from typing import Sequence

from geo_ph import Coxian2, fit_coxian2, ph_mean_var_from_stages


def coxian_mg1k_metrics(stage_moments: Sequence[tuple[float, float]], lambda_rate: float, k: int) -> dict[str, float]:
    m, v = ph_mean_var_from_stages(stage_moments)
    scv = v / (m * m)
    # Geometric mean (Wang Eq. 6)
    target = lambda_rate * m
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        s = sum(mid ** i for i in range(1, k + 1))
        if s < target:
            lo = mid
        else:
            hi = mid
    alpha = (lo + hi) / 2
    e_tx = alpha / (1 - alpha)
    e_wait = e_tx / lambda_rate
    # Coxian-2 fit
    ph = fit_coxian2(m, scv)
    return {
        "mean": m,
        "var": v,
        "scv": scv,
        "alpha": alpha,
        "E_tx": e_tx,
        "E_wait": e_wait,
        "E_sojourn_geom": e_wait + m,
        "coxian_lambda1": ph.lambda1,
        "coxian_lambda2": ph.lambda2,
        "coxian_p": ph.p,
        "coxian_mean": ph.mean(),
        "coxian_var": ph.var(),
    }
