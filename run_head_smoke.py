#!/usr/bin/env python3
from __future__ import annotations
import json, math, random, hashlib, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

import laya
from laya.common import build_sequence, QTYPES

Q={
    't':'choice',
    'ins':(
        'Using only the contemporaneous numerical evidence for candidate_A and candidate_B, '
        'which candidate is more likely to have the higher total return over the next 21 trading sessions? '
        'Ticker identity and date are hidden. Do not assume candidate_A or the existing Hybrid24 ordering is correct.'
    ),
    'crit':{
        'A':'candidate_A is more likely to have the higher next-21-session return',
        'B':'candidate_B is more likely to have the higher next-21-session return',
    }
}

def collate(items,pad):
    n=len(items); L=max(len(x['ids']) for x in items)
    ids=torch.full((n,L),pad,dtype=torch.long); att=torch.zeros((n,L),dtype=torch.long)
    mp=torch.zeros((n,2),dtype=torch.long); mm=torch.ones((n,2),dtype=torch.bool)
    y=torch.zeros((n,2),dtype=torch.float32); qt=torch.full((n,),QTYPES['choice'],dtype=torch.long)
    for i,x in enumerate(items):
        ids[i,:len(x['ids'])]=torch.tensor(x['ids']); att[i,:len(x['ids'])]=1
        mp[i]=torch.tensor(x['markers']); y[i]=torch.tensor(x['target'])
    return dict(input_ids=ids,attention_mask=att,marker_pos=mp,marker_mask=mm,target=y,qtype=qt)

def item(tok,state,label,max_len=1024,head_max_len=192):
    seq,m=build_sequence(tok,state,Q,max_len,head_max_len)
    if len(m)!=2: raise RuntimeError('marker mismatch')
    target=[.98,.02] if int(label)==1 else [.02,.98]
    return {'ids':seq,'markers':m,'target':target}

def fwd(model,b,detach=True):
    z,_=model(b['input_ids'],b['attention_mask'],b['marker_pos'],b['marker_mask'],b['qtype'],detach_encoder=detach)
    return z

def met(y,p):
    y=np.asarray(y,int); p=np.clip(np.asarray(p,float),1e-6,1-1e-6); pred=(p>=.5).astype(int)
    return {
        'n':int(len(y)), 'accuracy':float(accuracy_score(y,pred)),
        'auc':float(roc_auc_score(y,p)) if len(np.unique(y))>1 else None,
        'brier':float(brier_score_loss(y,p)), 'log_loss':float(log_loss(y,p,labels=[0,1])),
        'mean_p':float(p.mean()), 'base_rate':float(y.mean())
    }

def fit_temp(z,y):
    best=(1e9,1.0)
    for t in np.exp(np.linspace(math.log(.5),math.log(5),81)):
        p=1/(1+np.exp(-np.clip(np.asarray(z)/t,-40,40)))
        ll=log_loss(y,p,labels=[0,1])
        if ll<best[0]: best=(ll,float(t))
    return best[1]

def eval_pairs(model,tok,pairs,batch=4):
    its=[]; meta=[]
    for r in pairs:
        its.append(item(tok,r['orig_state'],r['y'])); meta.append((r['pair_id'],'o'))
        its.append(item(tok,r['swap_state'],1-r['y'])); meta.append((r['pair_id'],'s'))
    rows=[]; model.eval()
    with torch.no_grad():
        for k in range(0,len(its),batch):
            ch=its[k:k+batch]; z=fwd(model,collate(ch,tok.pad_token_id)).numpy()
            for j,a in enumerate(z): rows.append((*meta[k+j],float(a[0]-a[1])))
    d=pd.DataFrame(rows,columns=['pair_id','ori','z']).pivot(index='pair_id',columns='ori',values='z')
    idx=[r['pair_id'] for r in pairs]
    zo=d.loc[idx,'o'].to_numpy(); zs=d.loc[idx,'s'].to_numpy(); zsym=.5*(zo-zs)
    po=1/(1+np.exp(-np.clip(zo,-40,40))); ps=1/(1+np.exp(-np.clip(zs,-40,40)))
    return zsym,float(np.mean(np.abs(po+ps-1)))

def numeric_diff(state):
    a=state['candidate_A']; b=state['candidate_B']; out={}
    for k in sorted(set(a)&set(b)):
        try: out[k]=float(a[k])-float(b[k])
        except (TypeError,ValueError): pass
    return out

def main():
    req={}
    for line in Path('data/pilot_requests.jsonl').read_text().splitlines():
        x=json.loads(line); req[x['case_id']]=x['state']
    lab=pd.read_csv('data/pilot_labels.csv')
    originals=lab[lab.orientation.eq('original')]
    pairs=[]
    for _,r in originals.iterrows():
        pid=r.pair_id
        pairs.append({'pair_id':pid,'y':int(r.a_wins),'orig_state':req[pid+'-o'],'swap_state':req[pid+'-s']})
    pairs.sort(key=lambda r: hashlib.sha256(('laya-head-smoke-v1:'+r['pair_id']).encode()).hexdigest())
    train,cal,test=pairs[:45],pairs[45:55],pairs[55:70]
    assert len(train)==45 and len(cal)==10 and len(test)==15

    t0=time.time(); agent=laya.load('convaiinnovations/laya',device='cpu'); load_s=time.time()-t0
    model,tok=agent.model,agent.tok
    z0,sym0=eval_pairs(model,tok,test); ytest=np.array([r['y'] for r in test]); p0=1/(1+np.exp(-np.clip(z0,-40,40)))

    # Numeric logistic baseline from exactly the same anonymized state.
    Xtr=pd.DataFrame([numeric_diff(r['orig_state']) for r in train]); Xte=pd.DataFrame([numeric_diff(r['orig_state']) for r in test])
    Xte=Xte.reindex(columns=Xtr.columns)
    clf=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(C=.25,max_iter=2000,random_state=7))
    clf.fit(Xtr,[r['y'] for r in train]); plog=clf.predict_proba(Xte)[:,1]

    for p in model.encoder.parameters(): p.requires_grad=False
    for p in model.act_head.parameters(): p.requires_grad=False
    trp=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('encoder.') and not n.startswith('act_head.')]
    train_items=[]
    for r in train:
        train_items.append(item(tok,r['orig_state'],r['y'])); train_items.append(item(tok,r['swap_state'],1-r['y']))
    opt=torch.optim.AdamW(trp,lr=2.5e-4,weight_decay=.01)
    rng=random.Random(7); rng.shuffle(train_items); losses=[]; model.train(); model.encoder.eval(); model.act_head.eval(); t1=time.time()
    for k in range(0,len(train_items),4):
        b=collate(train_items[k:k+4],tok.pad_token_id); opt.zero_grad(set_to_none=True)
        z=fwd(model,b); loss=-(b['target']*torch.log_softmax(z,-1)).sum(-1).mean(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trp,1.0); opt.step(); losses.append(float(loss.detach()))
    train_s=time.time()-t1

    zc,symc=eval_pairs(model,tok,cal); temp=fit_temp(zc,np.array([r['y'] for r in cal]))
    z1,sym1=eval_pairs(model,tok,test); p1=1/(1+np.exp(-np.clip(z1/temp,-40,40)))
    out={
        'protocol':'Laya decision-head smoke v1; deterministic pair-level split; NOT temporal OOS',
        'split':{'train_pairs':45,'calibration_pairs':10,'test_pairs':15},
        'model_load_s':load_s,'head_train_s':train_s,'mean_train_loss':float(np.mean(losses)),
        'temperature':temp,'trainable_head_params':int(sum(p.numel() for p in trp)),
        'zero_shot':met(ytest,p0),'zero_shot_swap_symmetry_mae':sym0,
        'head_finetuned':met(ytest,p1),'head_finetuned_swap_symmetry_mae':sym1,
        'calibration_swap_symmetry_mae':symc,'logistic':met(ytest,plog),
        'test_pair_ids':[r['pair_id'] for r in test],
    }
    Path('results_smoke').mkdir(exist_ok=True)
    Path('results_smoke/report.json').write_text(json.dumps(out,indent=2)+'\n')
    pd.DataFrame({'pair_id':out['test_pair_ids'],'y':ytest,'p_zero':p0,'p_ft':p1,'p_logistic':plog}).to_csv('results_smoke/predictions.csv',index=False)
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()
