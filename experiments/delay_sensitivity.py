import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from geo_model import GeoConfig, example_config, geo_service_time, stage_times, queue_metrics


def replace_delay(cfg, cross_delay_ms):
    delay_ms = [[0.5 if i == j else cross_delay_ms for j in range(cfg.regions)] for i in range(cfg.regions)]
    return GeoConfig(cfg.nodes_per_region, cfg.bandwidth_bytes_s,
                     tuple(tuple(x / 1000 for x in row) for row in delay_ms),
                     cfg.processing_s_byte, cfg.client_region, cfg.message_bytes,
                     cfg.header_bytes, cfg.transaction_bytes, cfg.block_capacity)


def main():
    base = example_config()
    out = Path('results'); out.mkdir(exist_ok=True)
    rows=[]
    for delay_ms in [0, 5, 15, 30, 50, 80, 120, 180, 240, 300]:
        cfg = replace_delay(base, delay_ms)
        for mode in ["all_receiver_quorum", "client_visible_quorum"]:
            T = geo_service_time(cfg, mode)
            metrics = queue_metrics(T, 5.0, cfg.block_capacity)
            stages = stage_times(cfg, 0, mode)
            rows.append({'mode':mode, 'cross_delay_ms':delay_ms, **stages, **metrics})
    with (out/'delay_sensitivity.csv').open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print('wrote', out/'delay_sensitivity.csv')
    for row in rows:
        print(row['mode'], row['cross_delay_ms'], 'ms -> T', row['T'], 'throughput', row['throughput'], 'E_sojourn', row['E_sojourn'])

if __name__ == '__main__': main()
