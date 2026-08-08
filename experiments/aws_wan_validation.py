import csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from geo_model import GeoConfig, example_config, stage_times, queue_metrics
from geo_wan import aws_delay_matrix, lognormal_jitter_samples, median_jitter, p99_jitter


def cfg_with_aws_matrix(base):
    names = ("us-east-1", "us-west-1", "eu-west-1", "ap-east-1")
    m = aws_delay_matrix(names)
    return GeoConfig(base.nodes_per_region, base.bandwidth_bytes_s,
                     tuple(tuple(x for x in row) for row in m), base.processing_s_byte,
                     base.client_region, base.message_bytes, base.header_bytes,
                     base.transaction_bytes, base.block_capacity), names


def main():
    base = example_config()
    cfg, names = cfg_with_aws_matrix(base)
    out = Path("results"); out.mkdir(exist_ok=True)
    rows = []
    # 1) worst-primary deterministic stage breakdown
    primary = max(range(cfg.regions), key=lambda p: stage_times(cfg, p)["total"])
    stages = stage_times(cfg, primary)
    metrics = queue_metrics(stages["total"], 5.0, cfg.block_capacity)
    rows.append({"scenario": "aws_worst_primary_deterministic",
                 "primary_region": primary,
                 "jitter_sigma": 0.0,
                 "jitter_median": 0.0,
                 "jitter_p99": 0.0,
                 **{f"stage_{k}": v for k, v in stages.items()},
                 **metrics})
    # 2) jitter sweep using log-normal tails
    for sigma in (0.1, 0.2, 0.3, 0.5, 0.8):
        all_samples = []
        # sample 1000 jittered round times via deterministic mean per region + log-normal
        for p in range(cfg.regions):
            base_delay = stage_times(cfg, p)["total"]
            samples = lognormal_jitter_samples(base_delay, sigma=sigma, n=1000, seed=20260804 + p)
            all_samples.extend(samples)
        med = median_jitter(all_samples)
        p99 = p99_jitter(all_samples)
        mean_total = sum(all_samples) / len(all_samples)
        rows.append({"scenario": "aws_round_time_log_normal",
                     "primary_region": -1,
                     "jitter_sigma": sigma,
                     "jitter_median": med,
                     "jitter_p99": p99,
                     "stage_pre_prepare": float("nan"),
                     "stage_prepare": float("nan"),
                     "stage_commit": float("nan"),
                     "stage_reply": float("nan"),
                     "stage_total": mean_total,
                     "T": mean_total,
                     "mu": 1 / mean_total,
                     "alpha": float("nan"),
                     "E_tx": float("nan"),
                     "E_wait": float("nan"),
                     "E_sojourn": float("nan"),
                     "throughput": float("nan"),
                     "rho": float("nan")})
    path = out / "aws_wan_validation.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print("wrote", path, "rows", len(rows))
    for r in rows:
        print(r["scenario"], "sigma=", r["jitter_sigma"],
              "median=", round(r["jitter_median"], 4), "p99=", round(r["jitter_p99"], 4),
              "mean_total=", round(r["stage_total"], 4))


if __name__ == "__main__":
    main()
