import csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from geo_model import GeoConfig, example_config, geo_service_time, queue_metrics


def with_inter_bandwidth(cfg, mbps):
    b = tuple(tuple((1e9 if i == j else mbps * 1e6) for j in range(cfg.regions)) for i in range(cfg.regions))
    return GeoConfig(cfg.nodes_per_region, b, cfg.delay_s, cfg.processing_s_byte,
                     cfg.client_region, cfg.message_bytes, cfg.header_bytes,
                     cfg.transaction_bytes, cfg.block_capacity)


def main():
    base=example_config(); out=Path('results'); out.mkdir(exist_ok=True); rows=[]
    for mbps in [10, 25, 50, 100, 250, 500, 1000]:
        cfg=with_inter_bandwidth(base, mbps); T=geo_service_time(cfg)
        rows.append({'inter_bandwidth_MBs':mbps, **queue_metrics(T,5.0,cfg.block_capacity)})
    with (out/'bandwidth_sensitivity.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print('wrote',out/'bandwidth_sensitivity.csv')
    for r in rows: print(r['inter_bandwidth_MBs'],'MB/s -> T',r['T'],'throughput',r['throughput'],'E_sojourn',r['E_sojourn'])

if __name__=='__main__': main()
