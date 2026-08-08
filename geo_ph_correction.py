"""PH-aware Pollaczek-Khinchine style correction for the geo-PBFT model.

Strict Theorem 1 assumes exponential service time.  In reality, the
geo-PBFT round time has finite variance, and a moment-matched PH
distribution (Coxian-2) yields a more accurate estimate of E_sojourn.

The standard PK formula applies to M/G/1:
  E_W = rho * E[R] / (1 - rho) * (1 + scv) / 2

We adapt it for the M/G^(1,k)/1 batch-service queue heuristically:
  E_sojourn_PK = E_sojourn_geom - E_T + E_T * (1 + scv) / 2
where E_sojourn_geom is the geometric baseline (Wang Eq. 10).

This is a heuristic, not an exact M/G^(1,k)/1 result.  The paper should
document the approximation explicitly and note the conditions under
which it is accurate (heavy-traffic regime, k >> 1).
"""
from __future__ import annotations

import math
from typing import Sequence

from geo_ph import Coxian2, fit_coxian2, ph_mean_var_from_stages


def sojourn_pk_correction(geometric_e_sojourn: float, mean_T: float, scv: float) -> float:
    """Apply the Pollaczek-Khinchine style correction to the geometric baseline.

    This is a heuristic adapted from M/G/1 to M/G^(1,k)/1.  It assumes the
    heavy-traffic regime and k >> 1.
    """
    if mean_T <= 0:
        raise ValueError("mean_T must be positive")
    if scv < 0:
        raise ValueError("scv must be non-negative")
    pk_service_term = mean_T * (1 + scv) / 2
    return geometric_e_sojourn - mean_T + pk_service_term


def sojourn_with_ph(geometric_e_sojourn: float, stage_moments: Sequence[tuple[float, float]]) -> dict[str, float]:
    """Compute the PH-aware E_sojourn and the related Coxian-2 parameters."""
    m, v = ph_mean_var_from_stages(stage_moments)
    scv = v / (m * m)
    coxian = fit_coxian2(m, scv)
    return {
        "mean": m,
        "var": v,
        "scv": scv,
        "coxian_lambda1": coxian.lambda1,
        "coxian_lambda2": coxian.lambda2,
        "coxian_p": coxian.p,
        "E_sojourn_geom": geometric_e_sojourn,
        "E_sojourn_PK": sojourn_pk_correction(geometric_e_sojourn, m, scv),
        "delta": sojourn_pk_correction(geometric_e_sojourn, m, scv) - geometric_e_sojourn,
    }


if __name__ == "__main__":
    # 4 stages with realistic moments
    moments = [
        (0.18, 0.1),  # pre-prepare: mean 180 ms, SCV 0.1 (relatively deterministic)
        (0.18, 1.0),  # prepare: Exp
        (0.18, 1.0),  # commit: Exp
        (0.08, 1.0),  # reply: Exp
    ]
    geom_e_sojourn = 1.241  # from prior run
    result = sojourn_with_ph(geom_e_sojourn, moments)
    for k, v in result.items():
        print(f"  {k:25s} {v}")
