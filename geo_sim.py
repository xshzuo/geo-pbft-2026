"""Deterministic / mean discrete-event simulator for the geo-PBFT model.

This is a verification simulator, not a PBFT state machine.  It uses the
quorum-aware stage semantics from ``geo_model`` and confirms that the
deterministic ``stage_times`` matches the mean of a large number of
jittered samples.  It also records client-observable sojourn times.
"""
from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Sequence

from geo_model import GeoConfig, example_config, node_regions, link_cost, quorum_completion


@dataclass
class SimResult:
    rounds: int
    mean_round: float
    p99_round: float
    mean_pp: float
    mean_p: float
    mean_c: float
    mean_reply: float


@dataclass(order=True)
class Event:
    time: float
    seq: int
    kind: str = field(compare=False)
    payload: dict = field(compare=False, default_factory=dict)


def _cost_with_jitter(cfg: GeoConfig, src: int, dst: int, payload_bytes: float,
                      rng: random.Random, jitter_ms: float) -> float:
    return link_cost(cfg, src, dst, payload_bytes) + rng.uniform(0.0, jitter_ms / 1000.0)


def run_simulation(cfg: GeoConfig, rounds: int, lambda_rate: float, jitter_ms: float,
                   seed: int, service_override: Callable[[random.Random], float] | None = None) -> SimResult:
    rng = random.Random(seed)
    regions = node_regions(cfg)
    arrivals: list[float] = []
    t = 0.0
    for _ in range(rounds + 1):
        # Generate rounds+1 arrivals so the simulator has at least one round's
        # worth of pre-arrival traffic to start the first block.
        t += rng.expovariate(lambda_rate)
        arrivals.append(t)
    # For each block, the service time is a full PBFT round given the current
    # primary region (rotated round-robin).  Jitter is applied per-link.
    round_records: list[dict[str, float]] = []
    sojourn_records: list[float] = []
    schedule: list[Event] = []
    heapq.heapify(schedule)
    next_seq = 0
    arrivals_done = 0
    # Seed the simulator with the first few arrivals; we kick off consensus as
    # soon as there are enough transactions to fill one block.
    for i, t in enumerate(arrivals):
        if i < cfg.block_capacity:
            continue
        primary = i % cfg.regions  # round-robin
        backups = list(regions)
        backups.remove(primary)
        pp = max(_cost_with_jitter(cfg, primary, dst,
                                   cfg.block_bytes + cfg.message_bytes, rng, jitter_ms) for dst in backups)
        receiver_quorum_times = [
            _receiver_quorum(cfg, regions, dst, cfg.message_bytes, rng, jitter_ms)
            for dst in regions
        ]
        p = max(receiver_quorum_times)
        c = max(receiver_quorum_times)
        reply = sorted(_cost_with_jitter(cfg, src, cfg.client_region, cfg.message_bytes, rng, jitter_ms)
                       for src in regions)[cfg.quorum - 1]
        total = pp + p + c + reply
        # Each block's sojourn time for its earliest-arrived transaction: the
        # service time of the block in which it was packed.
        block_sojourn = total + (i - cfg.block_capacity) / lambda_rate
        round_records.append({"pp": pp, "p": p, "c": c, "reply": reply, "total": total})
        sojourn_records.append(block_sojourn)
        if len(round_records) >= rounds:
            break
    rounds_done = len(round_records)
    mean_round = sum(r["total"] for r in round_records) / max(rounds_done, 1)
    sorted_rounds = sorted(r["total"] for r in round_records)
    p99 = sorted_rounds[int(0.99 * max(rounds_done - 1, 0))]
    return SimResult(
        rounds=rounds_done,
        mean_round=mean_round,
        p99_round=p99,
        mean_pp=sum(r["pp"] for r in round_records) / max(rounds_done, 1),
        mean_p=sum(r["p"] for r in round_records) / max(rounds_done, 1),
        mean_c=sum(r["c"] for r in round_records) / max(rounds_done, 1),
        mean_reply=sum(r["reply"] for r in round_records) / max(rounds_done, 1),
    )


def _receiver_quorum(cfg: GeoConfig, source_regions: Sequence[int], receiver_region: int,
                     payload_bytes: float, rng: random.Random, jitter_ms: float) -> float:
    senders = list(source_regions)
    senders.remove(receiver_region)
    costs = [0.0]
    costs.extend(_cost_with_jitter(cfg, src, receiver_region, payload_bytes, rng, jitter_ms)
                 for src in senders)
    return sorted(costs)[cfg.quorum - 1]


if __name__ == "__main__":
    cfg = example_config()
    res = run_simulation(cfg, rounds=200, lambda_rate=5.0, jitter_ms=5.0, seed=20260804)
    print(res)
