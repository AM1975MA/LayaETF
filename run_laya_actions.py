#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,brier_score_loss,log_loss,roc_auc_score

QUESTION_TEXT=(
    "Using only the contemporaneous numerical evidence in candidate_A and candidate_B, "
    "which candidate is more likely to have the higher total return over the next 21 trading sessions? "
    "Higher rank and hybrid scores mean stronger model preference. Asset identities and dates are hidden. "
    "Do not assume the existing Hybrid24 ordering is correct."
)

def question(variant:str):
    if variant=='noul_neutral':
        return {'a_outperforms_b':{
            'type':'noul',
            'instructions':QUESTION_TEXT + ' Answer true if candidate_A is more likely to outperform candidate_B.',
            'criteria':{
                'true':'candidate_A is more likely to outperform candidate_B',
                'false':'candidate_B is at least as likely to outperform candidate_A',
            },
            'labels':{'true':'A','false':'B'},
        }}
    if variant=='choice':
        return {'winner':{
            'type':'choice',
            'instructions':QUESTION_TEXT,
            'criteria':{
                'A':'candidate_A is more likely to have the higher next-21-session total return',
                'B':'candidate_B is more likely to have the higher next-21-session total return',
            },
        }}
    raise ValueError(variant)

def p_a(res,variant):
    if variant=='noul_neutral': return float(res['answers']['a_outperforms_b']['noul'])
    return float(res['answers']['winner']['probabilities']['A'])

def ece(y,p,k=10):
    edges=np.linspace(0,1,k+1); bins=np.clip(np.digitize(p,edges,right=True)-1,0,k-1)
    total=0.; rows=[]
    for j in range(k):
        m=bins==j
        if not m.any(): continue
        mp=float(p[m].mean()); wr=float(y[m].mean()); n=int(m.sum())
        total += n/len(y)*abs(mp-wr)
        rows.append({'bin':j,'n':n,'mean_p':mp,'win_rate':wr})
    return float(total),rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--requests',default='data/pilot_requests.jsonl')
    ap.add_argument('--labels',default='data/pilot_labels.csv')
    ap.add_argument('--variant',choices=['noul_neutral','choice'],required=True)
    ap.add_argument('--model',default='convaiinnovations/laya')
    ap.add_argument('--batch-size',type=int,default=4)
    ap.add_argument('--max-len',type=int,default=1024)
    ap.add_argument('--head-max-len',type=int,default=192)
    ap.add_argument('--outdir',default='results')
    a=ap.parse_args()
    import laya
    states=[]; ids=[]
    with open(a.requests,encoding='utf-8') as f:
        for line in f:
            x=json.loads(line); ids.append(x['case_id']); states.append(x['state'])
    qs=question(a.variant)
    t0=time.perf_counter(); agent=laya.load(a.model,device='cpu'); load_s=time.perf_counter()-t0
    t1=time.perf_counter()
    if hasattr(agent,'predict_batch'):
        results=agent.predict_batch(
            states,qs,batch_size=a.batch_size,sort_by_length=True,
            max_len=a.max_len,head_max_len=a.head_max_len,
        )
    else:
        results=[agent.predict(s,qs,max_len=a.max_len,head_max_len=a.head_max_len) for s in states]
    infer_s=time.perf_counter()-t1
    pred=pd.DataFrame({'case_id':ids,'p_A':[p_a(r,a.variant) for r in results]})
    lab=pd.read_csv(a.labels)
    z=lab.merge(pred,on='case_id',how='inner',validate='one_to_one')
    o=z[z.orientation.eq('original')].copy(); y=o.a_wins.astype(int).to_numpy(); p=o.p_A.to_numpy(float)
    m={
        'variant':a.variant,'model':a.model,'n_original':int(len(o)),'n_requests':int(len(z)),
        'accuracy':float(accuracy_score(y,p>=.5)),
        'brier':float(brier_score_loss(y,p)),
        'log_loss':float(log_loss(y,p,labels=[0,1])),
        'auc':float(roc_auc_score(y,p)) if len(np.unique(y))>1 else None,
        'mean_p_A':float(p.mean()),'empirical_A_win_rate':float(y.mean()),
        'model_load_s':load_s,'inference_s':infer_s,'requests_per_s':float(len(z)/infer_s),
        'max_len':a.max_len,'head_max_len':a.head_max_len,
    }
    m['ece10'],cal=ece(y,p)
    piv=z.pivot(index='pair_id',columns='orientation',values='p_A').dropna()
    err=(piv['original']+piv['swapped']-1).abs()
    m['swap_pairs']=int(len(err)); m['swap_symmetry_mae']=float(err.mean()); m['swap_symmetry_p95']=float(err.quantile(.95))
    o['q']=pd.qcut(o.p_A.rank(method='first'),5,labels=False)+1
    q=o.groupby('q').agg(n=('a_wins','size'),mean_p=('p_A','mean'),win_rate=('a_wins','mean')).reset_index()
    out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    pred.to_csv(out/f'predictions_{a.variant}.csv',index=False)
    report={'metrics':m,'calibration_bins':cal,'probability_quintiles':q.to_dict(orient='records')}
    (out/f'report_{a.variant}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
