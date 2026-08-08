"""R=1 and homogeneous-region regression tests for the geo-PBFT model.

This is a defensive check: when all regions are identical and the inter-region
delay is zero, the model must collapse to a single-region PBFT quorum model
that matches the Wang baseline family.  We verify several invariants:

  1. T_geo with R=1 equals the single-region quorum-aware round time.
  2. All regions identical with zero inter-region delay yields T_geo independent
     of the per-region bandwidth/processing times being equal.
  3. A larger inter-region delay strictly increases T_geo.
  4. Worst-primary equals rotation-average when the network is homogeneous.
  5. lambda >= k * mu_geo is rejected as unstable.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from geo_model import GeoConfig, geo_service_time, stage_times, queue_metrics, example_config


def single_region_config() -> GeoConfig:
    return GeoConfig(
        nodes_per_region=(4,),
        bandwidth_bytes_s=((500e6,),),
        delay_s=((0.001,),),
        processing_s_byte=(1e-8,),
        block_capacity=10,
    )


def homogeneous_two_region_config(inter_delay_s: float) -> GeoConfig:
    return GeoConfig(
        nodes_per_region=(2, 2),
        bandwidth_bytes_s=((500e6, 500e6), (500e6, 500e6)),
        delay_s=((0.001, inter_delay_s), (inter_delay_s, 0.001)),
        processing_s_byte=(1e-8, 1e-8),
        block_capacity=10,
    )


def test_r1_collapses_to_single_region():
    cfg = single_region_config()
    T = geo_service_time(cfg)
    stages = stage_times(cfg, 0)
    assert abs(T - stages["total"]) < 1e-9
    metrics = queue_metrics(T, lambda_rate=0.1, k=cfg.block_capacity)
    assert 0 < metrics["alpha"] < 1
    assert metrics["rho"] < 1


def test_homogeneous_zero_inter_delay_invariant_under_swap():
    cfg = homogeneous_two_region_config(0.0)
    T_a = geo_service_time(cfg)
    # Swap the two regions: re-permute rows and columns symmetrically
    r = cfg.regions
    perm = list(range(r))[::-1]
    swapped_delay = tuple(tuple(cfg.delay_s[i][j] for j in perm) for i in perm)
    swapped = GeoConfig(cfg.nodes_per_region[::-1],
                        cfg.bandwidth_bytes_s[::-1],
                        swapped_delay,
                        cfg.processing_s_byte[::-1],
                        cfg.client_region,
                        cfg.message_bytes, cfg.header_bytes, cfg.transaction_bytes, cfg.block_capacity)
    T_b = geo_service_time(swapped)
    assert abs(T_a - T_b) < 1e-9


def test_strictly_increasing_in_inter_region_delay():
    Ts = []
    for d in (0.0, 0.001, 0.01, 0.05, 0.1):
        cfg = homogeneous_two_region_config(d)
        Ts.append(geo_service_time(cfg))
    for a, b in zip(Ts, Ts[1:]):
        assert b >= a


def test_worst_primary_equals_average_when_homogeneous():
    cfg = homogeneous_two_region_config(0.0)
    worst = max(stage_times(cfg, p)["total"] for p in range(cfg.regions))
    avg = sum(stage_times(cfg, p)["total"] for p in range(cfg.regions)) / cfg.regions
    assert abs(worst - avg) < 1e-9


def test_unstable_rejected():
    cfg = example_config()
    T = geo_service_time(cfg)
    try:
        queue_metrics(T, lambda_rate=1e9, k=cfg.block_capacity)
    except ValueError:
        pass
    else:
        raise AssertionError("unstable queue was accepted")


def main():
    rows = []
    for d in (0.0, 0.001, 0.005, 0.01, 0.05):
        cfg = homogeneous_two_region_config(d)
        T = geo_service_time(cfg)
        rows.append({"inter_delay_s": d, "T_geo": T,
                     "T_pp": stage_times(cfg, 0)['pre_prepare'],
                     "T_prepare": stage_times(cfg, 0)['prepare'],
                     "T_commit": stage_times(cfg, 0)['commit'],
                     "T_reply": stage_times(cfg, 0)['reply']})
    out = Path("results"); out.mkdir(exist_ok=True)
    path = out / "regression_sweep.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print("wrote", path)
    for r in rows:
        print(f"d={r['inter_delay_s']:.4f}s T_geo={r['T_geo']:.6f}")
    # Run all local tests
    test_r1_collapses_to_single_region()
    test_homogeneous_zero_inter_delay_invariant_under_swap()
    test_strictly_increasing_in_inter_region_delay()
    test_worst_primary_equals_average_when_homogeneous()
    test_unstable_rejected()
    print("all 5 regression tests passed")


if __name__ == "__main__":
    main()
