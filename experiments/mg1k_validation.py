import csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from geo_model import example_config, stage_times
from geo_mg1k import mg1k_batch_metrics


def worst_primary_stages(base):
    # use the worst primary region over a homogeneous WAN delay
    p = max(range(base.regions), key=lambda i: stage_times(base, i)['total'])
    return p, stage_times(base, p)


def main():
    cfg = example_config()
    out = Path("results"); out.mkdir(exist_ok=True)
    rows = []
    for delay in (0, 5, 15, 30, 80, 150, 240):
        from geo_model import GeoConfig
        r = cfg.regions
        delay_s = tuple(tuple((0.5 if i == j else delay) / 1000 for j in range(r)) for i in range(r))
        new_cfg = GeoConfig(cfg.nodes_per_region, cfg.bandwidth_bytes_s, delay_s,
                            cfg.processing_s_byte, cfg.client_region, cfg.message_bytes,
                            cfg.header_bytes, cfg.transaction_bytes, cfg.block_capacity)
        p, stages = worst_primary_stages(new_cfg)
        # Assume pre-prepare is more deterministic (SCV=0.1); other stages Exp (SCV=1)
        moments = [(stages['pre_prepare'], 0.1), (stages['prepare'], 1.0),
                   (stages['commit'], 1.0), (stages['reply'], 1.0)]
        metrics = mg1k_batch_metrics(moments, lambda_rate=5.0, k=cfg.block_capacity)
        rows.append({
            "cross_delay_ms": delay,
            "primary": p,
            **{f"stage_{k}": v for k, v in stages.items()},
            **metrics,
        })
    path = out / "mg1k_validation.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print("wrote", path, "rows", len(rows))
    for r in rows:
        print(f"delay={r['cross_delay_ms']:3}ms mean={r['mean']:.4f} scv={r['scv']:.3f} alpha={r['alpha']:.3f} "
              f"E_sojourn_geom={r['E_sojourn_geometric']:.4f} E_sojourn_PK={r['E_sojourn_PK']:.4f}")


if __name__ == "__main__":
    main()
