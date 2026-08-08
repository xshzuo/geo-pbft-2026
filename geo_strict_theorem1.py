"""Strict Theorem 1 statement using sample-path stochastic process.

This module provides a sample-path formulation of the M/M^(1,k)/1 queue
with quorum-aware service rate mu_geo.  The strict Theorem 1 is the
positive-recurrence criterion for the embedded Markov chain observed at
service completion epochs.

We also formalise the relationship:
  T_geo  =  deterministic / sample-mean round time
  mu_geo =  1 / T_geo
and provide the bootstrap confidence interval for the service-time
estimator.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Sequence


@dataclass
class StrictTheorem1:
    """Strict positive-recurrence criterion for the geo-PBFT M/M^(1,k)/1 queue."""
    lambda_rate: float
    mu_geo: float
    k: int

    def stable(self) -> bool:
        return self.lambda_rate < self.k * self.mu_geo

    def mean_sojourn_exponential(self) -> float:
        """E_sojourn under the exponential service-time assumption.

        Wang Eq. (10) with mu replaced by mu_geo:
          E_sojourn = alpha / (lambda * (1-alpha)) + 1 / mu_geo
        """
        if not self.stable():
            return float("inf")
        target = self.lambda_rate / self.mu_geo
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if sum(mid ** i for i in range(1, self.k + 1)) < target:
                lo = mid
            else:
                hi = mid
        alpha = (lo + hi) / 2
        e_wait = alpha / (self.lambda_rate * (1 - alpha))
        return e_wait + 1 / self.mu_geo


@dataclass
class SamplePathMuEstimate:
    """Sample-path estimator of mu_geo with bootstrap CI."""
    samples: Sequence[float]

    def mean(self) -> float:
        return sum(self.samples) / len(self.samples)

    def var(self) -> float:
        m = self.mean()
        return sum((x - m) ** 2 for x in self.samples) / max(len(self.samples) - 1, 1)

    def sd(self) -> float:
        return math.sqrt(self.var())

    def mu_hat(self) -> float:
        m = self.mean()
        if m <= 0:
            return float("inf")
        return 1.0 / m

    def mu_ci(self, alpha: float = 0.05, B: int = 200, seed: int = 20260804) -> tuple[float, float]:
        """Bootstrap CI on the mean; convert to CI on mu_geo = 1/mean."""
        import random
        rng = random.Random(seed)
        n = len(self.samples)
        means = []
        for _ in range(B):
            boot = [self.samples[rng.randrange(n)] for _ in range(n)]
            means.append(sum(boot) / n)
        means.sort()
        lo_mean = means[int(B * alpha / 2)]
        hi_mean = means[int(B * (1 - alpha / 2))]
        # 1/x is monotone-decreasing on positive reals, so the CI on
        # mu_geo is the reversed CI on the mean.
        return (1 / hi_mean, 1 / lo_mean)


def strict_theorem_1_statement() -> str:
    return (
        "Theorem 1 (Strict region-aware quorum-aware stability, sample-path form). "
        "Let T_n be the per-round service time of the geo-PBFT instance, with "
        "E[T_n] = T_geo (defined in Eq. 3.6) and Var[T_n] finite.  Define mu_geo = 1/T_geo. "
        "Let the arrival process be Poisson(lambda).  Then the embedded Markov chain "
        "observed at service-completion epochs is positive recurrent if and only if "
        "lambda < k * mu_geo.  In this regime the stationary queue-length distribution "
        "is geometric: pi_i = (1-alpha) alpha^i, where alpha is the unique root of "
        "sum_{i=1}^k alpha^i = lambda / mu_geo."
    )


if __name__ == "__main__":
    import random
    rng = random.Random(20260804)
    # Generate fake round-time samples around a target of T_geo = 0.6 s
    samples = [0.6 + rng.gauss(0, 0.05) for _ in range(500)]
    samples = [max(0.01, s) for s in samples]
    estimator = SamplePathMuEstimate(samples)
    print("strict T_n mean", estimator.mean(), "var", estimator.var())
    print("mu_hat = 1 / mean =", estimator.mu_hat())
    lo, hi = estimator.mu_ci()
    print(f"95% CI on mu_geo: [{lo:.4f}, {hi:.4f}]")
    thm = StrictTheorem1(lambda_rate=5.0, mu_geo=estimator.mu_hat(), k=100)
    print("lambda < k*mu_geo:", thm.stable())
    print("E_sojourn (exponential):", thm.mean_sojourn_exponential())
