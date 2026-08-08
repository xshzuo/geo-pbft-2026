import csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from geo_model import GeoConfig, example_config, geo_service_time, stage_times, queue_metrics

DELAYS_MS = [0, 5, 15, 30, 80, 150, 240]
BANDWIDTHS_MBS = [1, 5, 10, 50, 100, 500]


def config_for(base, delay_ms, bandwidth_mbs):
    r=base.regions
    delay=tuple(tuple((0.5 if i==j else delay_ms)/1000 for j in range(r)) for i in range(r))
    bandwidth=tuple(tuple(1e9 if i==j else bandwidth_mbs*1e6 for j in range(r)) for i in range(r))
    return GeoConfig(base.nodes_per_region,bandwidth,delay,base.processing_s_byte,
                     base.client_region,base.message_bytes,base.header_bytes,
                     base.transaction_bytes,base.block_capacity)


def main():
    base=example_config(); out=Path('results');out.mkdir(exist_ok=True);rows=[]
    for delay in DELAYS_MS:
        for bw in BANDWIDTHS_MBS:
            cfg=config_for(base,delay,bw)
            # use the worst-primary stage breakdown consistently
            primary=max(range(cfg.regions),key=lambda p:stage_times(cfg,p)['total'])
            stages=stage_times(cfg,primary)
            metrics=queue_metrics(stages['total'],5.0,cfg.block_capacity)
            dominant=max(('pre_prepare','prepare','commit','reply'),key=lambda x:stages[x])
            rows.append({'cross_delay_ms':delay,'inter_bandwidth_MBs':bw,
                         'worst_primary_region':primary,'dominant_stage':dominant,
                         **stages,**metrics})
    path=out/'delay_bandwidth_grid.csv'
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print('wrote',path,'rows',len(rows))
    for d in DELAYS_MS:
        subset=[r for r in rows if r['cross_delay_ms']==d]
        lo,hi=subset[0],subset[-1]
        print(f"delay={d:3}ms: T@1MB/s={lo['T']:.6f}, T@500MB/s={hi['T']:.6f}, dominant={hi['dominant_stage']}")

if __name__=='__main__':main()
