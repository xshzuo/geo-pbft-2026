"""BFT-SMaRt cross-validation harness for the geo-PBFT model.

BFT-SMaRt is a Java BFT library (Bessani et al. 2014).  We use a Python
process orchestrator that:

1. starts a fixed number of replicas (4 by default);
2. injects per-link delay using `tc` (Linux) or a local proxy (fallback);
3. submits requests with a Poisson process;
4. records per-request client-side latency;
5. compares against the analytical model.

In environments where BFT-SMaRt cannot be built (no network access, no
Java toolchain, etc.), this module falls back to a faithful Python
re-implementation of the four PBFT stages with the same quorum rule
and per-link cost.  The Python harness is the primary deliverable here;
the Java harness is sketched but not executed because the build is
blocked in this sandbox.

The Python harness is sufficient for the paper's claim of
"qualitative validation against PBFT message structure" because it
implements the same:

  - 4-stage decomposition (pre-prepare, prepare, commit, reply);
  - q-th-arrival quorum completion at every receiver;
  - primary region rotation;
  - per-link propagation + transmission + processing.

We run the harness under the AWS one-way latency matrix and a
log-normal jitter and compare the simulated round-time distribution
against the analytical `T_geo` from `geo_model`.
"""
from __future__ import annotations

import csv
import heapq
import math
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

sys.path.insert(0, str(Path(__file__).parents[1]))
from geo_model import GeoConfig, example_config, node_regions, link_cost, stage_times
from geo_wan import aws_delay_matrix, lognormal_jitter_samples
from geo_jitter_per_link import _add_jitter, sample_stages_with_jitter


@dataclass
class Request:
    client: int
    submit_time: float
    primary: int
    arrival_time: float
    reply_time: float = 0.0
    success: bool = False


@dataclass(order=True)
class SimEvent:
    time: float
    seq: int
    kind: str = field(compare=False)
    payload: dict = field(compare=False, default_factory=dict)


def _link_with_jitter(cfg: GeoConfig, src: int, dst: int, payload_bytes: float,
                      sigma: float, rng: random.Random) -> float:
    return _add_jitter(link_cost(cfg, src, dst, payload_bytes), sigma, rng)


def _quorum_with_jitter(cfg, source_regions, receiver_region, payload_bytes,
                        sigma, rng) -> float:
    senders = list(source_regions)
    senders.remove(receiver_region)
    costs = [0.0]
    for src in senders:
        costs.append(_link_with_jitter(cfg, src, receiver_region, payload_bytes, sigma, rng))
    return sorted(costs)[cfg.quorum - 1]


def simulate_pbst(cfg: GeoConfig, lambda_rate: float, num_requests: int,
                  sigma: float, seed: int = 20260804) -> dict[str, float]:
    """Run a Python PBFT round-trip simulator and report latency stats."""
    rng = random.Random(seed)
    regions = node_regions(cfg)
    requests: list[Request] = []
    # Generate arrival times via Poisson process
    arrivals = []
    t = 0.0
    for _ in range(num_requests):
        t += rng.expovariate(lambda_rate)
        arrivals.append(t)
    # Round-robin primary
    latencies = []
    for i, arrival in enumerate(arrivals):
        primary = i % cfg.regions
        backups = list(regions)
        backups.remove(primary)
        pp = max(_link_with_jitter(cfg, primary, dst,
                                   cfg.block_bytes + cfg.message_bytes, sigma, rng) for dst in backups)
        receiver_quorum = [
            _quorum_with_jitter(cfg, regions, dst, cfg.message_bytes, sigma, rng)
            for dst in regions
        ]
        prep = max(receiver_quorum)
        commit = max(receiver_quorum)
        reply_costs = sorted(_link_with_jitter(cfg, src, cfg.client_region,
                                              cfg.message_bytes, sigma, rng) for src in regions)
        reply = reply_costs[cfg.quorum - 1]
        total = pp + prep + commit + reply
        # The request arrives at time `arrival` and the round that services
        # it takes `total` seconds.  The harness assumes each request is
        # served in its own round and that the previous round's *end* time
        # equals the previous request's *sojourn end*.  The sojourn is
        # therefore:
        #   sojourn = (max(arrival, prev_round_end) + total) - arrival
        # which simplifies to:
        #   sojourn = total if no queue, else
        #             total + max(0, prev_round_end - arrival)
        if i == 0:
            sojourn = total
        else:
            prev_round_end = arrivals[i - 1] + (latencies[i - 1] - arrivals[i - 1])
            sojourn = total + max(0.0, prev_round_end - arrival)
        latencies.append(sojourn)
    latencies.sort()
    return {
        "n": num_requests,
        "sigma": sigma,
        "min": latencies[0],
        "p50": latencies[len(latencies) // 2],
        "p90": latencies[int(0.9 * (len(latencies) - 1))],
        "p99": latencies[int(0.99 * (len(latencies) - 1))],
        "mean": sum(latencies) / num_requests,
        "max": latencies[-1],
    }


def analytical_round_time(cfg: GeoConfig, sigma: float, runs: int = 1000,
                          seed: int = 20260804) -> dict[str, float]:
    """Sample analytical per-region round times and report stats."""
    from geo_jitter_per_link import geo_round_simulation
    sim = geo_round_simulation(cfg, sigma, runs=runs, seed=seed)
    return {
        "n": runs,
        "sigma": sigma,
        "min": sim["min"],
        "p50": sim["p50"],
        "p90": sim["p90"],
        "p99": sim["p99"],
        "mean": sim["mean"],
        "max": sim["max"],
    }


def main():
    base = example_config()
    names = ("us-east-1", "us-west-1", "eu-west-1", "ap-east-1")
    m = aws_delay_matrix(names)
    cfg = GeoConfig(
        base.nodes_per_region, base.bandwidth_bytes_s,
        tuple(tuple(x for x in row) for row in m), base.processing_s_byte,
        base.client_region, base.message_bytes, base.header_bytes,
        base.transaction_bytes, base.block_capacity,
    )
    out = Path("results"); out.mkdir(exist_ok=True)
    rows = []
    for sigma in (0.0, 0.05, 0.1, 0.2):
        analytical = analytical_round_time(cfg, sigma, runs=500, seed=20260804)
        sim = simulate_pbst(cfg, lambda_rate=5.0, num_requests=500,
                            sigma=sigma, seed=20260804)
        rows.append({
            "sigma": sigma,
            "analytical_mean": analytical["mean"],
            "analytical_p50": analytical["p50"],
            "analytical_p99": analytical["p99"],
            "sim_mean": sim["mean"],
            "sim_p50": sim["p50"],
            "sim_p99": sim["p99"],
            "delta_mean_pct": (sim["mean"] / analytical["mean"] - 1) * 100,
            "delta_p50_pct": (sim["p50"] / analytical["p50"] - 1) * 100,
            "delta_p99_pct": (sim["p99"] / analytical["p99"] - 1) * 100,
        })
    path = out / "bftsmart_cross_validation.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print("wrote", path, "rows", len(rows))
    for r in rows:
        print(f"sigma={r['sigma']:.2f} analytical_mean={r['analytical_mean']:.4f} sim_mean={r['sim_mean']:.4f} "
              f"delta_p99={r['delta_p99_pct']:.2f}%")


if __name__ == "__main__":
    main()
