import csv
import math
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from geo_model import GeoConfig, example_config, node_regions, link_cost, stage_times, geo_service_time, queue_metrics

RNG_SEED = 20260804
DELAYS_MS = [0, 5, 15, 30, 80, 150, 240]
JITTER_MS = [0, 1, 5, 10, 25, 50]
RUNS = 30


def cfg_with(base: GeoConfig, delay_ms: float, jitter_ms: float) -> GeoConfig:
    r = base.regions
    base_delay = tuple(tuple((0.5 if i == j else delay_ms) / 1000 for j in range(r)) for i in range(r))
    return GeoConfig(
        nodes_per_region=base.nodes_per_region,
        bandwidth_bytes_s=base.bandwidth_bytes_s,
        delay_s=base_delay,
        processing_s_byte=base.processing_s_byte,
        client_region=base.client_region,
        message_bytes=base.message_bytes,
        header_bytes=base.header_bytes,
        transaction_bytes=base.transaction_bytes,
        block_capacity=base.block_capacity,
    )


def sample_rounds(cfg: GeoConfig, runs: int, jitter_ms: float, seed: int) -> dict[str, list[float]]:
    rng = random.Random(seed)
    times: dict[str, list[float]] = {"pre_prepare": [], "prepare": [], "commit": [], "reply": [], "total": []}
    for _ in range(runs):
        # Pick primary region uniformly at random; jitter on every one-way link
        primary = rng.randrange(cfg.regions)
        regions = node_regions(cfg)
        backups = list(regions)
        backups.remove(primary)
        pp = max(link_cost(cfg, primary, dst, cfg.block_bytes + cfg.message_bytes)
                 + rng.uniform(0, jitter_ms / 1000) for dst in backups)
        prep = max(quorum_completion_jitter(cfg, regions, dst, cfg.message_bytes, rng, jitter_ms)
                   for dst in regions)
        comm = max(quorum_completion_jitter(cfg, regions, dst, cfg.message_bytes, rng, jitter_ms)
                   for dst in regions)
        reply_costs = sorted(
            link_cost(cfg, src, cfg.client_region, cfg.message_bytes) + rng.uniform(0, jitter_ms / 1000)
            for src in regions
        )
        reply = reply_costs[cfg.quorum - 1]
        times["pre_prepare"].append(pp)
        times["prepare"].append(prep)
        times["commit"].append(comm)
        times["reply"].append(reply)
        times["total"].append(pp + prep + comm + reply)
    return times


def quorum_completion_jitter(cfg, source_regions, receiver_region, payload_bytes, rng, jitter_ms):
    senders = list(source_regions)
    senders.remove(receiver_region)
    costs = [0.0]
    for src in senders:
        costs.append(link_cost(cfg, src, receiver_region, payload_bytes) + rng.uniform(0, jitter_ms / 1000))
    return sorted(costs)[cfg.quorum - 1]


def main():
    base = example_config()
    out = Path("results"); out.mkdir(exist_ok=True)
    rows = []
    deterministic_baseline = {d: geo_service_time(cfg_with(base, d, 0)) for d in DELAYS_MS}
    for delay_ms in DELAYS_MS:
        for jitter_ms in JITTER_MS:
            cfg = cfg_with(base, delay_ms, jitter_ms)
            aggregate: dict[str, list[float]] = {"pre_prepare": [], "prepare": [], "commit": [], "reply": [], "total": []}
            for seed in range(RUNS):
                run = sample_rounds(cfg, 1, jitter_ms, RNG_SEED + seed)
                for k, v in run.items():
                    aggregate[k].append(v[0])
            mean_total = sum(aggregate["total"]) / RUNS
            p99 = sorted(aggregate["total"])[int(0.99 * RUNS) - 1]
            metrics = queue_metrics(mean_total, 5.0, cfg.block_capacity)
            metrics_det = queue_metrics(deterministic_baseline[delay_ms], 5.0, cfg.block_capacity)
            rows.append({
                "cross_delay_ms": delay_ms,
                "jitter_ms": jitter_ms,
                "mean_total": mean_total,
                "deterministic_total": deterministic_baseline[delay_ms],
                "p99_total": p99,
                "p99_overhead_pct": (p99 / deterministic_baseline[delay_ms] - 1) * 100,
                "mean_overhead_pct": (mean_total / deterministic_baseline[delay_ms] - 1) * 100,
                "deterministic_throughput": metrics_det["throughput"],
                "jitter_throughput": metrics["throughput"],
                "deterministic_alpha": metrics_det["alpha"],
                "jitter_alpha": metrics["alpha"],
            })
    path = out / "jitter_sensitivity.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print("wrote", path, "rows", len(rows))
    for r in rows:
        print(f"delay={r['cross_delay_ms']:3}ms jitter={r['jitter_ms']:2}ms "
              f"mean={r['mean_total']:.4f} p99={r['p99_total']:.4f} "
              f"overhead_mean={r['mean_overhead_pct']:.2f}% "
              f"p99={r['p99_overhead_pct']:.2f}%")


if __name__ == "__main__":
    main()
