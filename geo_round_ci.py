"""Confidence-interval analysis for the round-time mean from per-link Monte-Carlo.

Given the per-link log-normal jitter from ``geo_jitter_per_link``, this
module computes the mean and a non-parametric confidence interval.  The
result is a single scalar (mu_hat) with a 95% confidence interval.

This is the "E[mu_geo] with confidence interval" piece required by
Section 3 v0 critique item 1.
"""
from __future__ import annotations

import math
import random
import statistics
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from geo_model import GeoConfig, example_config
from geo_wan import aws_delay_matrix
from geo_jitter_per_link import geo_round_simulation


def _bootstrap_ci(samples: list[float], alpha: float = 0.05, B: int = 200, seed: int = 20260804) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(samples)
    if n == 0:
        return float("nan"), float("nan")
    means = []
    for _ in range(B):
        boot = [samples[rng.randrange(n)] for _ in range(n)]
        means.append(sum(boot) / n)
    means.sort()
    lo = means[int(B * alpha / 2)]
    hi = means[int(B * (1 - alpha / 2))]
    return lo, hi


def geo_round_with_ci(cfg: GeoConfig, sigma: float, runs: int, seed: int = 20260804) -> dict[str, float]:
    sim = geo_round_simulation(cfg, sigma, runs=runs, seed=seed)
    totals = sorted(sim["p50"] for _ in [0])  # placeholder
    # Re-run with seed to obtain the raw samples directly via geo_round_simulation
    rng = random.Random(seed)
    raw = []
    for _ in range(runs):
        primary = rng.randrange(cfg.regions)
        from geo_jitter_per_link import sample_stages_with_jitter
        raw.append(sample_stages_with_jitter(cfg, primary, sigma, rng)["total"])
    mean = sum(raw) / runs
    sd = statistics.pstdev(raw) if runs > 1 else 0.0
    sem = sd / math.sqrt(runs)
    # Normal approximation CI
    z = 1.96
    ci_lo_n = mean - z * sem
    ci_hi_n = mean + z * sem
    # Bootstrap CI
    b_lo, b_hi = _bootstrap_ci(raw, alpha=0.05, B=300, seed=seed + 1)
    return {
        "sigma": sigma,
        "runs": runs,
        "mean": mean,
        "sd": sd,
        "sem": sem,
        "ci_normal_lo": ci_lo_n,
        "ci_normal_hi": ci_hi_n,
        "ci_bootstrap_lo": b_lo,
        "ci_bootstrap_hi": b_hi,
    }


def cfg_with_aws(base):
    names = ("us-east-1", "us-west-1", "eu-west-1", "ap-east-1")
    m = aws_delay_matrix(names)
    return GeoConfig(
        base.nodes_per_region, base.bandwidth_bytes_s,
        tuple(tuple(x for x in row) for row in m), base.processing_s_byte,
        base.client_region, base.message_bytes, base.header_bytes,
        base.transaction_bytes, base.block_capacity,
    )


def main():
    import csv
    base = example_config()
    cfg = cfg_with_aws(base)
    out = Path("results"); out.mkdir(exist_ok=True)
    rows = []
    for sigma in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5):
        row = geo_round_with_ci(cfg, sigma, 800)
        rows.append(row)
    path = out / "round_time_ci.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print("wrote", path, "rows", len(rows))
    for r in rows:
        print(f"sigma={r['sigma']:.2f} runs={r['runs']:4} mean={r['mean']:.4f} "
              f"sd={r['sd']:.4f} 95%CI=[{r['ci_bootstrap_lo']:.4f}, {r['ci_bootstrap_hi']:.4f}]")


if __name__ == "__main__":
    main()
