"""View-change Markov chain and steady-state availability for geo-PBFT.

This module computes the steady-state probability that the system is
in the NORMAL (serving) state vs in the VIEW-CHANGE state for a single
PBFT primary.  It also reports the view-change interruption factor
that feeds into ``geo_viewchange.py``.

The state diagram (per primary region) is:

  NORMAL  --lambda_vc-->  VIEW-CHANGE
  VIEW-CHANGE  --mu_vc-->  NORMAL

where:
  lambda_vc = expected view-change rate (Poisson)
  mu_vc     = 1 / T_vc (view-change completion rate)

This is a 2-state CTMC.  The stationary distribution is
  P(NORMAL)        = mu_vc / (lambda_vc + mu_vc)
  P(VIEW-CHANGE)   = lambda_vc / (lambda_vc + mu_vc)
"""
from __future__ import annotations

import math


def view_change_steady_state(lambda_vc: float, T_vc: float) -> dict[str, float]:
    if lambda_vc < 0:
        raise ValueError("invalid parameters")
    if T_vc == 0:
        return {"lambda_vc": lambda_vc, "T_vc": T_vc, "mu_vc": float("inf"),
                "p_normal": 1.0, "p_vc": 0.0, "interruption_factor": 1.0}
    mu_vc = 1.0 / T_vc
    p_normal = mu_vc / (lambda_vc + mu_vc)
    p_vc = lambda_vc / (lambda_vc + mu_vc)
    # Effective service rate under interruption:
    # mu_eff = (fraction of time in NORMAL) * mu_n
    # where mu_n is the normal service rate of the system.
    # The expected service time per round is:
    #   T_with_vc = T_normal + (lambda_vc * T_vc) * T_normal
    #             = T_normal * (1 + lambda_vc * T_vc)
    # This matches the interruption factor used in ``geo_viewchange.py``.
    return {
        "lambda_vc": lambda_vc,
        "T_vc": T_vc,
        "mu_vc": mu_vc,
        "p_normal": p_normal,
        "p_vc": p_vc,
        "interruption_factor": 1 + lambda_vc * T_vc,
    }


def stability_with_vc(lambda_rate: float, mu_n: float, lambda_vc: float, T_vc: float) -> dict[str, float]:
    """Return effective mu and stability margin under view-change interruption.

    The effective service time per round is ``T_n * (1 + lambda_vc * T_vc)``.
    Stability requires ``lambda * T_eff < 1``, i.e. ``lambda < mu_n / (1 + lambda_vc * T_vc)``.
    """
    if lambda_rate <= 0 or mu_n <= 0 or lambda_vc < 0 or T_vc < 0:
        raise ValueError("invalid parameters")
    factor = 1 + lambda_vc * T_vc
    mu_eff = mu_n / factor
    margin = mu_eff - lambda_rate
    return {
        "lambda_rate": lambda_rate,
        "mu_n": mu_n,
        "lambda_vc": lambda_vc,
        "T_vc": T_vc,
        "factor": factor,
        "mu_eff": mu_eff,
        "stability_margin": margin,
        "stable": margin > 0,
    }


if __name__ == "__main__":
    for lv, tv in ((0.0, 0.0), (0.01, 1.0), (0.05, 1.0), (0.1, 2.0), (0.2, 3.0)):
        ss = view_change_steady_state(lv, tv)
        print(f"lambda_vc={lv:.2f} T_vc={tv:.2f}s  P_normal={ss['p_normal']:.4f} "
              f"P_vc={ss['p_vc']:.4f} factor={ss['interruption_factor']:.3f}")
    print()
    mu_n = 1.0 / 0.6407
    for lv, tv in ((0.0, 0.0), (0.1, 2.0), (0.5, 1.0)):
        s = stability_with_vc(1.0, mu_n, lv, tv)
        print(f"lambda=1.0 mu_n={mu_n:.3f} lambda_vc={lv} T_vc={tv}s  "
              f"mu_eff={s['mu_eff']:.4f} margin={s['stability_margin']:.4f} stable={s['stable']}")
