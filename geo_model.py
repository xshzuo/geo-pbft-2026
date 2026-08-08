"""Quorum-aware timing model for geo-distributed PBFT.

This is a research scaffold, not a production PBFT implementation.  It makes
one explicit approximation: each stage completes when the slowest receiver
obtains its required quorum of messages.  Pairwise link costs are deterministic
and include propagation, serialization, and receiver-side processing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import math


@dataclass(frozen=True)
class GeoConfig:
    nodes_per_region: tuple[int, ...]
    bandwidth_bytes_s: tuple[tuple[float, ...], ...]
    delay_s: tuple[tuple[float, ...], ...]
    processing_s_byte: tuple[float, ...]
    client_region: int = 0
    message_bytes: float = 150.0
    header_bytes: float = 80.0
    transaction_bytes: float = 300.0
    block_capacity: int = 100
    beta: float = 1.0

    @property
    def regions(self) -> int:
        return len(self.nodes_per_region)

    @property
    def n(self) -> int:
        return sum(self.nodes_per_region)

    @property
    def f(self) -> int:
        if self.n < 4 or self.n != 3 * ((self.n - 1) // 3) + 1:
            raise ValueError("base model requires N = 3f + 1")
        return (self.n - 1) // 3

    @property
    def quorum(self) -> int:
        return 2 * self.f + 1

    @property
    def block_bytes(self) -> float:
        return self.header_bytes + self.block_capacity * self.transaction_bytes

    def validate(self) -> None:
        r = self.regions
        if r == 0 or any(x < 1 for x in self.nodes_per_region):
            raise ValueError("every region must contain at least one node")
        if len(self.bandwidth_bytes_s) != r or len(self.delay_s) != r:
            raise ValueError("network matrices must be R x R")
        if len(self.processing_s_byte) != r:
            raise ValueError("processing vector must have length R")
        for matrix in (self.bandwidth_bytes_s, self.delay_s):
            if any(len(row) != r for row in matrix):
                raise ValueError("network matrices must be square")
        if not (0 <= self.client_region < r):
            raise ValueError("invalid client region")
        if self.block_capacity < 1:
            raise ValueError("block capacity must be positive")


def node_regions(cfg: GeoConfig) -> list[int]:
    return [r for r, count in enumerate(cfg.nodes_per_region) for _ in range(count)]


def link_cost(cfg: GeoConfig, src_region: int, dst_region: int, payload_bytes: float) -> float:
    """One-way deterministic cost from a source region to a destination."""
    return cfg.delay_s[src_region][dst_region] + payload_bytes / cfg.bandwidth_bytes_s[src_region][dst_region] + cfg.processing_s_byte[dst_region] * payload_bytes


def quorum_completion(cfg: GeoConfig, source_regions: Sequence[int], receiver_region: int,
                      payload_bytes: float, include_local: bool = True) -> float:
    """Time for a receiver in receiver_region to obtain its quorum.

    ``source_regions`` contains one entry per replica. When ``include_local``
    is true, exactly one receiver-local replica is removed from the network
    senders and replaced by one zero-cost local vote. This prevents counting
    the receiver twice.
    """
    senders = list(source_regions)
    costs: list[float]
    if include_local:
        try:
            senders.remove(receiver_region)
        except ValueError as exc:
            raise ValueError("receiver region has no local replica") from exc
        costs = [0.0]
    else:
        costs = []
    costs.extend(link_cost(cfg, src, receiver_region, payload_bytes) for src in senders)
    q = cfg.quorum
    if len(costs) < q:
        raise ValueError("not enough messages to form PBFT quorum")
    return sorted(costs)[q - 1]


def stage_times(cfg: GeoConfig, primary_region: int, completion_mode: str = "all_receiver_quorum") -> dict[str, float]:
    """Compute quorum-aware completion time for one primary region.

    ``all_receiver_quorum`` is conservative: every receiver must obtain its
    quorum. ``client_visible_quorum`` is a lower-bound sensitivity variant:
    only the q fastest receiver-side quorum completions are required. The
    latter is an explicit approximation for client-observable completion and
    must not be presented as the exact PBFT commit rule without protocol-level
    validation.
    """
    cfg.validate()
    if completion_mode not in {"all_receiver_quorum", "client_visible_quorum"}:
        raise ValueError("unknown completion_mode")
    regions = node_regions(cfg)
    # The model currently tracks a primary region rather than a primary node.
    # Remove exactly one replica from that region; excluding every node in the
    # primary's region would incorrectly erase intra-region pre-prepare traffic.
    backups = list(regions)
    backups.remove(primary_region)
    pp_costs = [
        link_cost(cfg, primary_region, dst, cfg.block_bytes + cfg.message_bytes)
        for dst in backups
    ]
    pp = max(pp_costs, default=0.0)
    receiver_quorum_times = [
        quorum_completion(cfg, regions, dst, cfg.message_bytes) for dst in regions
    ]
    if completion_mode == "all_receiver_quorum":
        p = max(receiver_quorum_times)
        c = max(receiver_quorum_times)
    else:
        # Sensitivity approximation: the q fastest receivers are sufficient
        # for a client-visible certificate. This is intentionally separate
        # from the exact PBFT state machine.
        p = sorted(receiver_quorum_times)[cfg.quorum - 1]
        c = sorted(receiver_quorum_times)[cfg.quorum - 1]
    reply_costs = sorted(link_cost(cfg, src, cfg.client_region, cfg.message_bytes) for src in regions)
    reply = reply_costs[cfg.quorum - 1]
    return {"pre_prepare": pp, "prepare": p, "commit": c, "reply": reply,
            "total": pp + p + c + reply}


def geo_service_time(cfg: GeoConfig, completion_mode: str = "all_receiver_quorum") -> float:
    """Conservative service time: worst primary-region rotation."""
    return max(stage_times(cfg, p, completion_mode)["total"] for p in range(cfg.regions))


def wang_service_time(cfg: GeoConfig, bandwidth: float | None = None,
                      gamma: float | None = None, beta: float = .25) -> float:
    """Wang et al. Eq. (2), with the same block/message parameters."""
    b = bandwidth if bandwidth is not None else sum(sum(x) for x in cfg.bandwidth_bytes_s) / (cfg.regions ** 2)
    g = gamma if gamma is not None else sum(cfg.processing_s_byte) / cfg.regions
    n, m, C = cfg.n, cfg.message_bytes, cfg.block_bytes
    return 2 * beta * m * n * n / b + (C + 2 * m) * n / b + g * (C + 2 * m * n + m)


def alpha_from_lambda(lambda_rate: float, mu: float, k: int, tol: float = 1e-12) -> float:
    """Unique alpha in (0,1) solving sum(alpha**i, i=1..k)=lambda/mu."""
    if lambda_rate <= 0 or mu <= 0 or lambda_rate >= k * mu:
        raise ValueError("unstable or invalid queue")
    lo, hi = 0.0, 1.0
    target = lambda_rate / mu
    for _ in range(200):
        mid = (lo + hi) / 2
        value = sum(mid ** i for i in range(1, k + 1))
        if value < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2


def queue_metrics(service_time: float, lambda_rate: float, k: int) -> dict[str, float]:
    mu = 1.0 / service_time
    alpha = alpha_from_lambda(lambda_rate, mu, k)
    e_tx = alpha / (1 - alpha)
    e_wait = e_tx / lambda_rate
    return {"T": service_time, "mu": mu, "alpha": alpha, "E_tx": e_tx,
            "E_wait": e_wait, "E_sojourn": e_wait + service_time,
            "throughput": k * mu, "rho": lambda_rate / (k * mu)}


def example_config() -> GeoConfig:
    # Four regions; 100 nodes satisfies N=3f+1 (f=33).
    nodes = (30, 25, 25, 20)
    bandwidth = tuple(tuple(500e6 if i != j else 1e9 for j in range(4)) for i in range(4))
    delay_ms = ((0.5, 15, 80, 180), (15, 0.5, 100, 170),
                (80, 100, 0.5, 200), (180, 170, 200, 0.5))
    delay = tuple(tuple(x / 1000 for x in row) for row in delay_ms)
    return GeoConfig(nodes, bandwidth, delay, (10e-9, 10e-9, 20e-9, 20e-9))


if __name__ == "__main__":
    cfg = example_config()
    geo_t = geo_service_time(cfg)
    wang_t = wang_service_time(cfg)
    print("N=", cfg.n, "f=", cfg.f, "quorum=", cfg.quorum)
    print("Wang T=%.6fs, Geo quorum-aware T=%.6fs" % (wang_t, geo_t))
    print("Geo metrics:", queue_metrics(geo_t, 5.0, cfg.block_capacity))
