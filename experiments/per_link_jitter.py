import csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from geo_model import example_config, geo_service_time, stage_times, GeoConfig
from geo_wan import aws_delay_matrix
from geo_jitter_per_link import geo_round_simulation
from geo_mg1k import mg1k_batch_metrics


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
    base = example_config()
    cfg = cfg_with_aws(base)
    out = Path("results"); out.mkdir(exist_ok=True)
    rows = []
    det_T = geo_service_time(cfg)
    rows.append({"sigma": 0.0, "method": "deterministic_worst_primary",
                 "mean": det_T, "p50": det_T, "p99": det_T, "ph_E_sojourn_geom": float("nan"),
                 "ph_E_sojourn_PK": float("nan")})
    for sigma in (0.05, 0.1, 0.2, 0.3, 0.5):
        sim = geo_round_simulation(cfg, sigma, runs=2000, seed=20260804)
        ph = mg1k_batch_metrics([(sim['mean'], 0.3)], 5.0, cfg.block_capacity)
        rows.append({"sigma": sigma, "method": "per_link_log_normal",
                     "mean": sim['mean'], "p50": sim['p50'], "p99": sim['p99'],
                     "ph_E_sojourn_geom": ph['E_sojourn_geometric'],
                     "ph_E_sojourn_PK": ph['E_sojourn_PK']})
    path = out / "per_link_jitter.csv"
    fieldnames = ["sigma", "method", "mean", "p50", "p99", "ph_E_sojourn_geom", "ph_E_sojourn_PK"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(rows)
    print("wrote", path, "rows", len(rows))
    for r in rows:
        print(f"sigma={r['sigma']:.2f} method={r['method']} mean={r['mean']:.4f} "
              f"p50={r.get('p50', 'n/a')} p99={r.get('p99', 'n/a')}")


if __name__ == "__main__":
    main()
