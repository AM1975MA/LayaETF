#!/usr/bin/env python3
import json, glob
from pathlib import Path
import numpy as np

reports=[]
for p in sorted(glob.glob('collected/**/report.json',recursive=True)):
    reports.append(json.load(open(p)))
if not reports:
    raise SystemExit('no reports found')
models=['laya_head_ft','logistic','xgboost']
summary={'folds':[r['fold'] for r in reports],'models':{}}
for m in models:
    rows=[r['metrics'][m] for r in reports]
    summary['models'][m]={k:float(np.mean([x[k] for x in rows])) for k in ['accuracy','auc','brier','log_loss','ece10']}
summary['hybrid24_top1_accuracy']=float(np.mean([r['metrics']['hybrid24_top1']['accuracy'] for r in reports]))
la=summary['models']['laya_head_ft']; lg=summary['models']['logistic']; xg=summary['models']['xgboost']
summary['pre_registered_signal_gate']=bool(la['auc']>0.52 and la['auc']>max(lg['auc'],xg['auc']) and la['brier']<min(lg['brier'],xg['brier']))
summary['fold_signal_checks']={str(r['fold']):bool(r['signal_check']) for r in reports}
Path('aggregate').mkdir(exist_ok=True)
Path('aggregate/summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
