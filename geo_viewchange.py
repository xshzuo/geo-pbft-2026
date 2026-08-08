"""View-change as a service interruption in the geo-PBFT queue.

Models the system as alternating phases:
  * normal service with rate mu_n
  * view-change interruption with rate gamma_vc and mean duration T_vc

Returns a heuristic correction to E_sojourn.  This is an M/G/1 with
vacations approximation, not an exact result; the paper should
document the approximation explicitly.
"""
from __future__ import annotations

import math


def view_change_correction(lambda_rate: float, mu_n: float, gamma_vc: float, T_vc: float) -> dict[str, float]:
    """Approximate expected sojourn under view-change interruptions.

    Treat view-change as a Bernoulli interruption: between any two normal
    services, with rate ``gamma_vc`` the system goes into a view-change of
    expected length ``T_vc``.  The effective mean service time is
    ``1/mu_n + gamma_vc * T_vc / mu_n`` and we use a simple M/M/1-style
    correction for the sojourn.
    """
    if lambda_rate <= 0 or mu_n <= 0 or gamma_vc < 0 or T_vc < 0:
        raise ValueError("invalid parameters")
    interruption_factor = 1 + gamma_vc * T_vc
    mean_service_with_vc = (1 / mu_n) * interruption_factor
    mu_eff = mu_n / interruption_factor
    if lambda_rate >= mu_n:
        return {"mu_eff": mu_n, "alpha": float("nan"), "rho": float("inf"),
                "E_sojourn": float("inf"), "interruption_factor": interruption_factor,
                "mean_service_with_vc": mean_service_with_vc,
                "note": "normal service alone is unstable; cannot complete correction"}
    rho_eff = lambda_rate * mean_service_with_vc
    if rho_eff >= 1:
        return {"mu_eff": mu_eff, "alpha": float("nan"), "rho": rho_eff,
                "E_sojourn": float("inf"), "interruption_factor": interruption_factor,
                "mean_service_with_vc": mean_service_with_vc,
                "note": "interrupted system is unstable"}
    return {
        "mu_eff": mu_eff,
        "rho": rho_eff,
        "E_sojourn": 1 / (mu_eff - lambda_rate),
        "interruption_factor": interruption_factor,
        "mean_service_with_vc": mean_service_with_vc,
    }
