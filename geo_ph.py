"""Phase-type service time utility for the geo-PBFT model.

Given a list of stages with mean and squared-coefficient-of-variation
(SCV), build an equivalent PH (Erlang-mix) representation.  The
representation is the standard hyperexponential/Erlang-mixture approach
used in queueing theory: fit a Coxian-2 by moment matching.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass
class Coxian2:
    """Two-stage Coxian distribution (a.k.a. C_2).

    The density is a mixture of two exponentials: Exp(lambda1) with
    probability p, and Exp(lambda2) (after passing through Exp(lambda1)
    with probability 1-p).
    """
    lambda1: float
    lambda2: float
    p: float

    def mean(self) -> float:
        return 1.0 / self.lambda1 + (1.0 - self.p) / self.lambda2

    def var(self) -> float:
        first = 1.0 / (self.lambda1 ** 2)
        second = 2.0 * (1.0 - self.p) / (self.lambda2 ** 2) + (1.0 - self.p) / (self.lambda1 * self.lambda2)
        return first + second


def fit_coxian2(target_mean: float, target_scv: float) -> Coxian2:
    """Match the first two moments of an arbitrary positive random variable
    with a Coxian-2 distribution.

    ``target_scv`` is the squared coefficient of variation.  When
    ``target_scv <= 1`` we use an Erlang-mixture by fixing p=0.5.
    When ``target_scv > 1`` we use a hyperexponential mixture.
    """
    if target_mean <= 0:
        raise ValueError("target_mean must be positive")
    if target_scv <= 0:
        raise ValueError("target_scv must be positive")
    var = target_scv * target_mean ** 2
    if abs(target_scv - 1.0) < 1e-9:
        # Pure exponential
        return Coxian2(lambda1=1.0 / target_mean, lambda2=1.0 / target_mean, p=1.0)
    if target_scv < 1.0:
        # Erlang-mixture: two stages, both with same rate
        # E[X] = 2/lambda, Var[X] = 2/lambda^2 => SCV = 0.5
        # We need SCV between 0.5 and 1.  Use a mix: p=0.5, two rates.
        p = 0.5
        # Two equations: 1/lambda1 + 0.5/lambda2 = m
        # 1/lambda1^2 + 0.75/lambda2^2 = var
        # Solve numerically
        a, b = _solve_coxian(target_mean, var, p)
        return Coxian2(lambda1=a, lambda2=b, p=p)
    # Hyperexponential case: p1 = p, rates lambda1, lambda2.
    # E[X] = p/lambda1 + (1-p)/lambda2
    # Var[X] = 2p/lambda1^2 + 2(1-p)/lambda2^2 - (E[X])^2
    p = 0.5
    a, b = _solve_hyperexponential(target_mean, var, p)
    return Coxian2(lambda1=a, lambda2=b, p=p)


def _solve_coxian(mean: float, var: float, p: float) -> tuple[float, float]:
    # Set a = 1/lambda1, b = 1/lambda2.
    # a + 0.5 b = m
    # a^2 + 0.75 b^2 = v
    # From first: a = m - 0.5 b
    # Substitute: (m - 0.5 b)^2 + 0.75 b^2 = v
    # m^2 - m b + 0.25 b^2 + 0.75 b^2 = v
    # m^2 - m b + b^2 = v
    # b^2 - m b + (m^2 - v) = 0
    a_term = 1.0
    b_term = -mean
    c_term = mean * mean - var
    disc = b_term * b_term - 4 * a_term * c_term
    if disc < 0:
        # Var too small for SCV<1 fit; clamp
        disc = 0
    b = (-b_term - math.sqrt(disc)) / (2 * a_term)
    if b <= 0:
        b = mean  # fallback
    a = mean - 0.5 * b
    if a <= 0:
        a = mean * 0.5
    return 1.0 / a, 1.0 / b


def _solve_hyperexponential(mean: float, var: float, p: float) -> tuple[float, float]:
    # E[X] = p/lambda1 + (1-p)/lambda2 = m
    # Var[X] = 2p/lambda1^2 + 2(1-p)/lambda2^2 - m^2
    # We need the rates to be positive.  Set lambda1 = lambda2 * r,
    # iterate to find r that gives the desired SCV.
    def scv_for_r(r: float) -> float:
        # Solve p/r/lambda2 + (1-p)/lambda2 = m => lambda2 = 1/m
        lambda2 = 1.0 / mean
        lambda1 = lambda2 * r
        e = p / lambda1 + (1 - p) / lambda2
        v = 2 * p / lambda1 ** 2 + 2 * (1 - p) / lambda2 ** 2 - e * e
        return v / (e * e)
    # If p = 0.5 and lambda1 = lambda2, SCV = 1
    # To get SCV > 1, spread the two rates.  Use r in (1, 20].
    lo, hi = 1.0, 50.0
    target_scv = var / (mean * mean)
    for _ in range(100):
        mid = (lo + hi) / 2
        s = scv_for_r(mid)
        if s < target_scv:
            lo = mid
        else:
            hi = mid
    r = (lo + hi) / 2
    lambda2 = 1.0 / mean
    return lambda2 * r, lambda2


def ph_mean_var_from_stages(stages: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """Sum independent stage moments.  Each entry is (mean, scv)."""
    m = 0.0
    v = 0.0
    for mean, scv in stages:
        m += mean
        v += scv * mean * mean
    return m, v


if __name__ == "__main__":
    # demo: target 4 stages each Exp(0.01) -> overall mean ~0.04, var ~0.0016
    stages = [(0.01, 1.0)] * 4
    m, v = ph_mean_var_from_stages(stages)
    print("sum mean", m, "sum var", v, "scv", v / (m * m))
    ph = fit_coxian2(m, v / (m * m))
    print("Coxian-2 mean", ph.mean(), "var", ph.var(), "scv", ph.var() / (ph.mean() ** 2))
