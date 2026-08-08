import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from geo_model import example_config, stage_times, geo_service_time, queue_metrics

out = Path('results')
out.mkdir(exist_ok=True)
cfg = example_config()
rows=[]
for primary in range(cfg.regions):
    st = stage_times(cfg, primary)
    row={'primary_region':primary, **st}
    rows.append(row)

T=geo_service_time(cfg)
rows.append({'primary_region':'worst_case', **stage_times(cfg, max(range(cfg.regions), key=lambda p: stage_times(cfg,p)['total']))})
with (out/'initial_primary_rotation.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)

metrics=queue_metrics(T,5.0,cfg.block_capacity)
with (out/'initial_queue_metrics.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=metrics.keys()); writer.writeheader(); writer.writerow(metrics)
print('wrote', out/'initial_primary_rotation.csv')
print('wrote', out/'initial_queue_metrics.csv')
print('geo_T',T,'metrics',metrics)
