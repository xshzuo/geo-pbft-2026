import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from geo_model import example_config, geo_service_time
from geo_viewchange import view_change_correction


def main():
    cfg = example_config()
    T = geo_service_time(cfg)
    mu_n = 1 / T
    print(f"normal mu_n = {mu_n:.4f} blocks/s, baseline T = {T:.4f} s")
    print("View-change sweep (lambda=5, mu_n=1/T):")
    for gamma, T_vc in ((0.0, 0.0), (0.01, 0.5), (0.05, 1.0), (0.1, 2.0), (0.2, 3.0)):
        r = view_change_correction(5.0, mu_n, gamma, T_vc)
        print(f"  gamma={gamma:.2f} T_vc={T_vc:.2f}s  mu_eff={r['mu_eff']:.4f} "
              f"alpha={r['alpha']:.3f} E_sojourn={r['E_sojourn']:.4f} factor={r['interruption_factor']:.3f}")


if __name__ == "__main__":
    main()
    # Demonstrate a stable parameter set
    cfg = example_config()
    T = geo_service_time(cfg)
    # Use a much higher lambda so the system saturates and the correction
    # actually has effect; otherwise baseline mu_n is large enough that
    # the queue is far from saturation and view-change barely matters.
    print("Stable parameter sweep (lambda=1.0, mu_n=1/T):")
    for gamma, T_vc_ms in ((0.1, 200), (0.5, 1000), (1.0, 2000), (2.0, 3000)):
        r = view_change_correction(1.0, 1 / T, gamma, T_vc_ms / 1000)
        print(f"  gamma={gamma} T_vc={T_vc_ms}ms  mu_eff={r['mu_eff']:.4f} "
              f"rho={r['rho']:.3f} E_sojourn={r['E_sojourn']:.4f} factor={r['interruption_factor']:.3f}")
