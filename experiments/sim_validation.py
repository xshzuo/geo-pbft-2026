import csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from geo_model import example_config, geo_service_time, stage_times
from geo_sim import run_simulation


def main():
    cfg = example_config()
    out = Path("results"); out.mkdir(exist_ok=True)
    rows = []
    for jitter in (0, 1, 5, 10, 25, 50):
        for seed in (20260804, 123456, 42):
            res = run_simulation(cfg, rounds=400, lambda_rate=5.0, jitter_ms=jitter, seed=seed)
            rows.append({
                "jitter_ms": jitter,
                "seed": seed,
                "rounds": res.rounds,
                "mean_round": res.mean_round,
                "p99_round": res.p99_round,
                "mean_pp": res.mean_pp,
                "mean_p": res.mean_p,
                "mean_c": res.mean_c,
                "mean_reply": res.mean_reply,
            })
    path = out / "sim_validation.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print("wrote", path, "rows", len(rows))
    # deterministic baseline for comparison
    T = geo_service_time(cfg)
    print("deterministic T =", T)
    for r in rows:
        print(f"jitter={r['jitter_ms']:3} seed={r['seed']} mean={r['mean_round']:.6f} p99={r['p99_round']:.6f}")


if __name__ == "__main__":
    main()
