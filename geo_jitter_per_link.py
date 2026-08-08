"""Per-region log-normal jitter for the geo-PBFT model.

The previous AWS sweep sampled jitter on top of the deterministic
worst-primary total.  This module samples jitter independently on each
per-link delay.  This is closer to how RIPE Atlas and actual WAN
jitter behave: each link has its own distribution.
"""
from __future__ import annotations

import math
import random
from typing import Sequence

from geo_model import GeoConfig, node_regions, link_cost, quorum_completion, stage_times


def _add_jitter(value: float, sigma: float, rng: random.Random) -> float:
    if sigma <= 0:
        return value
    u1, u2 = rng.random(), rng.random()
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    delta = math.exp(sigma * z) - 1.0
    return max(0.0, value + delta)


def sample_stages_with_jitter(cfg: GeoConfig, primary_region: int, sigma: float,
                              rng: random.Random) -> dict[str, float]:
    regions = node_regions(cfg)
    backups = list(regions)
    backups.remove(primary_region)
    pp = max(_add_jitter(link_cost(cfg, primary_region, dst,
                                   cfg.block_bytes + cfg.message_bytes), sigma, rng) for dst in backups)
    receiver_quorum_times = [
        _quorum_with_jitter(cfg, regions, dst, cfg.message_bytes, sigma, rng)
        for dst in regions
    ]
    p = max(receiver_quorum_times)
    c = max(receiver_quorum_times)
    reply_costs = sorted(_add_jitter(link_cost(cfg, src, cfg.client_region,
                                               cfg.message_bytes), sigma, rng) for src in regions)
    reply = reply_costs[cfg.quorum - 1]
    return {"pre_prepare": pp, "prepare": p, "commit": c, "reply": reply,
            "total": pp + p + c + reply}


def _quorum_with_jitter(cfg, source_regions, receiver_region, payload_bytes, sigma, rng):
    senders = list(source_regions)
    senders.remove(receiver_region)
    costs = [0.0]
    costs.extend(_add_jitter(link_cost(cfg, src, receiver_region, payload_bytes), sigma, rng)
                 for src in senders)
    return sorted(costs)[cfg.quorum - 1]


def geo_round_simulation(cfg: GeoConfig, sigma: float, runs: int, seed: int) -> dict[str, float]:
    """Run ``runs`` Monte-Carlo rounds with random primary region rotation."""
    rng = random.Random(seed)
    totals = []
    for _ in range(runs):
        primary = rng.randrange(cfg.regions)
        totals.append(sample_stages_with_jitter(cfg, primary, sigma, rng)["total"])
    totals.sort()
    return {
        "n": runs,
        "sigma": sigma,
        "min": totals[0],
        "p50": totals[len(totals) // 2],
        "p90": totals[int(0.9 * (len(totals) - 1))],
        "p99": totals[int(0.99 * (len(totals) - 1))],
        "mean": sum(totals) / runs,
        "max": totals[-1],
    }
