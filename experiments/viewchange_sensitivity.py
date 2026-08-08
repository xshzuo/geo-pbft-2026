import csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from geo_model import example_config, geo_service_time
from geo_viewchange import view_change_correction


def main():
    cfg = example_config()
    T = geo_service_time(cfg)
    mu_n = 1 / T
    out = Path("results"); out.mkdir(exist_ok=True)
    rows = []
    for gamma in (0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5):
        for T_vc_ms in (50, 200, 500, 1000, 2000, 3000, 5000):
            r = view_change_correction(5.0, mu_n, gamma, T_vc_ms / 1000)
            rows.append({
                "gamma_vc": gamma,
                "T_vc_ms": T_vc_ms,
                "lambda_rate": 5.0,
                "mu_n": mu_n,
                **r,
            })
    path = out / "viewchange_sensitivity.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print("wrote", path, "rows", len(rows))
    for r in rows:
        print(f"gamma={r['gamma_vc']:.3f} T_vc={r['T_vc_ms']:5}ms "
              f"mu_eff={r['mu_eff']:.4f} rho={r['rho']:.3f} E_sojourn={r['E_sojourn']:.4f} factor={r['interruption_factor']:.3f}")


if __name__ == "__main__":
    main()
