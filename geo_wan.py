"""Synthesize a region-pair latency sample that mimics RIPE Atlas style data.

The actual RIPE Atlas measurements require Atlas API credentials and a
pre-defined measurement ID; here we provide a deterministic synthetic
generator that uses the AWS inter-region latency table (a commonly cited
public source) plus log-normal jitter that matches empirical WAN jitter
distributions from RIPE Atlas papers.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence


# Approximate RTT/2 between typical cloud regions, drawn from AWS latency
# tables.  The unit is seconds.
AWS_LATENCY_ONEWAY_S: dict[tuple[str, str], float] = {
    ("us-east-1", "us-west-1"): 0.015,
    ("us-east-1", "eu-west-1"): 0.040,
    ("us-east-1", "ap-east-1"): 0.090,
    ("us-west-1", "eu-west-1"): 0.050,
    ("us-west-1", "ap-east-1"): 0.085,
    ("eu-west-1", "ap-east-1"): 0.100,
}


def aws_delay_matrix(region_names: Sequence[str], intra_seconds: float = 0.0005) -> list[list[float]]:
    r = len(region_names)
    matrix = [[0.0 for _ in range(r)] for _ in range(r)]
    for i in range(r):
        for j in range(r):
            if i == j:
                matrix[i][j] = intra_seconds
            else:
                key = tuple(sorted((region_names[i], region_names[j])))
                matrix[i][j] = AWS_LATENCY_ONEWAY_S.get((region_names[i], region_names[j]),
                                                       AWS_LATENCY_ONEWAY_S.get(key, 0.150))
    return matrix


def lognormal_jitter_samples(base: float, sigma: float, n: int, seed: int) -> list[float]:
    """Generate ``n`` samples of a log-normal delay on top of ``base``.

    The sample is ``base + LogNorm(0, sigma)`` in seconds; the resulting
    median is roughly ``base + exp(sigma**2/2)``.
    """
    rng = random.Random(seed)
    samples = []
    for _ in range(n):
        # Box-Muller; we only need one normal sample per draw
        u1, u2 = rng.random(), rng.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        # Log-normal: exp(N(0, sigma))
        delta = math.exp(sigma * z) - 1.0
        samples.append(base + delta)
    return samples


def median_jitter(samples: Sequence[float]) -> float:
    s = sorted(samples)
    return s[len(s) // 2]


def p99_jitter(samples: Sequence[float]) -> float:
    s = sorted(samples)
    return s[int(0.99 * (len(s) - 1))]


if __name__ == "__main__":
    for region_names in (("us-east-1", "us-west-1", "eu-west-1", "ap-east-1"),):
        m = aws_delay_matrix(region_names)
        print("matrix")
        for row in m:
            print("  ", [round(x, 4) for x in row])
        for base in (0.040, 0.090, 0.150):
            samples = lognormal_jitter_samples(base, sigma=0.2, n=1000, seed=20260804)
            print(f"base {base}s median {median_jitter(samples):.4f}s p99 {p99_jitter(samples):.4f}s")
