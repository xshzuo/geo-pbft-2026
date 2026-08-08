"""M/G^(1,k)/1 batch-service queueing approximation.

Implements:
  * moments matching of a generic G with PH (Coxian-2) representation
  * approximate geometric stationary distribution ``pi_i = (1-alpha) alpha^i``
    with ``alpha`` determined by the moment-matched PH, then corrected
    against the deterministic mean.
  * M/G^(1,k)/1 sanity metrics (utilisation, throughput, mean sojourn).
"""
from __future__ import annotations

import math
from typing import Sequence

from geo_ph import Coxian2, fit_coxian2, ph_mean_var_from_stages


def alpha_geometric_batch(lambda_rate: float, mean: float, k: int) -> float:
    """Wang et al. (2026) Eq. (6) for the geometric distribution."""
    if lambda_rate <= 0 or mean <= 0 or k < 1:
        raise ValueError("invalid parameters")
    if lambda_rate >= k / mean:
        raise ValueError("unstable queue")
    target = lambda_rate * mean
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        s = sum(mid ** i for i in range(1, k + 1))
        if s < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def mg1k_batch_metrics(stage_moments: Sequence[tuple[float, float]], lambda_rate: float, k: int) -> dict[str, float]:
    """Approximate the M/G^(1,k)/1 batch queue using the moment-matched PH.

    This is a heuristic: we use the geometric stationary distribution as a
    first-order approximation, then scale ``E_sojourn`` using the SCV of
    the total service time via a Pollaczek-Khinchine style correction.
    """
    mean, var = ph_mean_var_from_stages(stage_moments)
    scv = var / (mean * mean)
    alpha = alpha_geometric_batch(lambda_rate, mean, k)
    e_tx = alpha / (1 - alpha)
    e_wait = e_tx / lambda_rate
    # PK-style correction factor: 1 + scv
    corrected_sojourn = e_wait + mean * (1 + scv) / 2
    throughput = k * lambda_rate / alpha  # mean service rate = lambda_rate / alpha
    return {
        "mean": mean,
        "var": var,
        "scv": scv,
        "alpha": alpha,
        "E_tx": e_tx,
        "E_wait": e_wait,
        "E_sojourn_geometric": e_wait + mean,
        "E_sojourn_PK": corrected_sojourn,
        "throughput": throughput,
        "rho": lambda_rate * mean / k,
    }


def coxian_from_stages(stage_moments: Sequence[tuple[float, float]]) -> Coxian2:
    m, v = ph_mean_var_from_stages(stage_moments)
    return fit_coxian2(m, v / (m * m))
