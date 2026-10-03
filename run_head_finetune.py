#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, os, random, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from xgboost import XGBClassifier

import laya
from laya.common import build_sequence, QTYPES

FEATURES = [
    'h24','base','tail','tail1','tail2','et','xgb','dis','r21','r63','mom','acc',
    'vol','dd','eff','r2','cb','ub'
]
QUESTION_INTERNAL = {
    't': 'choice',
    'ins': (
        'Using only the contemporaneous numerical evidence for candidate_A and candidate_B, '
        'which candidate is more likely to have the higher total return over the next 21 trading sessions? '
        'Higher h24/base/tail/et/xgb values represent stronger contemporaneous model preference. '
        'Ticker identity and date are hidden. Do not assume candidate_A or the existing Hybrid24 ordering is correct.'
    ),
    'crit': {
        'A': 'candidate_A is more likely to have the higher next-21-session return',
        'B': 'candidate_B is more likely to have the higher next-21-session return',
    },
}


def fmt(v):
    if pd.isna(v): return 'NA'
    return f'{float(v):.5f}'


def state_text(row, swap=False):
    a, b = ('B','A') if swap else ('A','B')
    def line(label, suffix):
        return label + ': ' + ' '.join(f'{f}={fmt(row[f+"_"+suffix])}' for f in FEATURES)
    return line('candidate_A', a) + '\n' + line('candidate_B', b)


def build_item(tok, cfg, row, swap=False, max_len=512, head_max_len=160):
    seq, markers = build_sequence(tok, state_text(row, swap), QUESTION_INTERNAL, max_len, head_max_len)
    if len(markers) != 2:
        raise RuntimeError(f'expected 2 markers, got {len(markers)}')
    y = int(row['label'])
    if swap: y = 1-y
    target = [0.98,0.02] if y == 1 else [0.02,0.98]
    return {'ids':seq,'markers':markers,'qtype':QTYPES['choice'],'target':target,'label':y}


def collate(items, pad_id):
    n=len(items); L=max(len(x['ids']) for x in items)
    ids=torch.full((n,L),pad_id,dtype=torch.long)
    att=torch.zeros((n,L),dtype=torch.long)
    mpos=torch.zeros((n,2),dtype=torch.long)
    mmask=torch.ones((n,2),dtype=torch.bool)
    target=torch.zeros((n,2),dtype=torch.float32)
    qtype=torch.full((n,),QTYPES['choice'],dtype=torch.long)
    for i,it in enumerate(items):
        ids[i,:len(it['ids'])]=torch.tensor(it['ids'])
        att[i,:len(it['ids'])]=1
        mpos[i]=torch.tensor(it['markers'])
        target[i]=torch.tensor(it['target'])
    return dict(input_ids=ids,attention_mask=att,marker_pos=mpos,marker_mask=mmask,target=target,qtype=qtype)


def forward(model,b):
    logits,_=model(b['input_ids'],b['attention_mask'],b['marker_pos'],b['marker_mask'],b['qtype'],detach_encoder=True)
    return logits


def metrics(y,p):
    y=np.asarray(y,int); p=np.clip(np.asarray(p,float),1e-6,1-1e-6)
    pred=(p>=.5).astype(int)
    out={
        'n':int(len(y)),
        'accuracy':float(accuracy_score(y,pred)),
        'auc':float(roc_auc_score(y,p)) if len(np.unique(y))>1 else None,
        'brier':float(brier_score_loss(y,p)),
        'log_loss':float(log_loss(y,p,labels=[0,1])),
        'mean_p':float(p.mean()),
        'base_rate':float(y.mean()),
    }
    edges=np.linspace(0,1,11); bi=np.clip(np.digitize(p,edges,right=True)-1,0,9); ece=0.0
    for j in range(10):
        m=bi==j
        if m.any(): ece += m.mean()*abs(float(p[m].mean())-float(y[m].mean()))
    out['ece10']=float(ece)
    conf=np.abs(p-.5)
    order=pd.Series(conf).rank(method='first')
    q=pd.qcut(order,5,labels=False,duplicates='drop')
    bins=[]
    for j in sorted(pd.unique(q)):
        m=np.asarray(q==j)
        bins.append({'q':int(j)+1,'n':int(m.sum()),'mean_conf':float(conf[m].mean()),'accuracy':float((pred[m]==y[m]).mean())})
    out['confidence_quintiles']=bins
    return out


def fit_temperature(z,y):
    z=np.asarray(z,float); y=np.asarray(y,int)
    grid=np.exp(np.linspace(math.log(.5),math.log(5.0),81))
    best=(1e99,1.0)
    for t in grid:
        p=1/(1+np.exp(-np.clip(z/t,-40,40)))
        ll=log_loss(y,p,labels=[0,1])
        if ll<best[0]: best=(ll,float(t))
    return best[1]


def eval_laya(model,tok,cfg,df,batch_size,max_len=512,head_max_len=160):
    all_items=[]; meta=[]
    for idx,row in df.iterrows():
        for sw in (False,True):
            all_items.append(build_item(tok,cfg,row,sw,max_len,head_max_len)); meta.append((idx,sw))
    model.eval(); rows=[]
    with torch.no_grad():
        for s in range(0,len(all_items),batch_size):
            ch=all_items[s:s+batch_size]; b=collate(ch,tok.pad_token_id)
            z=forward(model,b).float().cpu().numpy()
            for j,zz in enumerate(z):
                idx,sw=meta[s+j]; rows.append((idx,sw,float(zz[0]-zz[1])))
    pred=pd.DataFrame(rows,columns=['idx','swap','z'])
    piv=pred.pivot(index='idx',columns='swap',values='z')
    z_orig=piv[False].reindex(df.index).to_numpy()
    z_swap=piv[True].reindex(df.index).to_numpy()
    z_sym=0.5*(z_orig-z_swap)
    po=1/(1+np.exp(-np.clip(z_orig,-40,40))); ps=1/(1+np.exp(-np.clip(z_swap,-40,40)))
    symmetry=float(np.mean(np.abs(po+ps-1)))
    return z_sym, symmetry


def baseline_matrix(df):
    X=pd.DataFrame(index=df.index)
    X['margin']=df['margin']
    for f in FEATURES:
        X['d_'+f]=df[f+'_A']-df[f+'_B']
    return X


def load_data(data_dir):
    frames=[]
    for p in sorted(Path(data_dir).glob('dev_20*.csv')):
        frames.append(pd.read_csv(p))
    if not frames: raise RuntimeError('no dev_*.csv files')
    return pd.concat(frames,ignore_index=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--fold',type=int,required=True,choices=[2020,2021,2022])
    ap.add_argument('--data-dir',default='data/finetune')
    ap.add_argument('--outdir',default='results_head')
    ap.add_argument('--epochs',type=int,default=1)
    ap.add_argument('--batch-size',type=int,default=4)
    ap.add_argument('--seed',type=int,default=26072026)
    args=ap.parse_args()
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','4')))
    random.seed(args.seed+args.fold); np.random.seed((args.seed+args.fold)%2**32); torch.manual_seed(args.seed+args.fold)

    df=load_data(args.data_dir)
    cal_year=args.fold-1
    train=df[df.year <= args.fold-2].copy()
    calib=df[(df.year==cal_year) & (df.exit_year < args.fold)].copy()
    test=df[df.year==args.fold].copy()
    if min(len(train),len(calib),len(test))==0: raise RuntimeError('empty split')

    Xtr=baseline_matrix(train); Xte=baseline_matrix(test)
    ytr=train.label.astype(int); yte=test.label.astype(int).to_numpy()
    logit=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(C=.25,max_iter=2000,random_state=args.seed))
    logit.fit(Xtr,ytr); p_log=logit.predict_proba(Xte)[:,1]
    xgb=XGBClassifier(n_estimators=160,max_depth=3,learning_rate=.035,subsample=.8,colsample_bytree=.8,
                      min_child_weight=8,reg_lambda=2.0,reg_alpha=.1,n_jobs=4,random_state=args.seed,eval_metric='logloss')
    xgb.fit(Xtr,ytr); p_xgb=xgb.predict_proba(Xte)[:,1]

    t0=time.time(); agent=laya.load('convaiinnovations/laya',device='cpu'); load_s=time.time()-t0
    model,tok,cfg=agent.model,agent.tok,agent.cfg
    for p in model.encoder.parameters(): p.requires_grad=False
    for p in model.act_head.parameters(): p.requires_grad=False
    trainable=[]
    for name,p in model.named_parameters():
        if p.requires_grad and not name.startswith('encoder.') and not name.startswith('act_head.'):
            trainable.append(p)
    n_trainable=sum(p.numel() for p in trainable)
    model.train(); model.encoder.eval(); model.act_head.eval()

    train_items=[]
    for _,row in train.iterrows():
        train_items.append(build_item(tok,cfg,row,False)); train_items.append(build_item(tok,cfg,row,True))
    opt=torch.optim.AdamW(trainable,lr=2.5e-4,weight_decay=.01)
    losses=[]; train_start=time.time()
    for ep in range(args.epochs):
        rng=random.Random(args.seed+args.fold+ep); rng.shuffle(train_items)
        ep_loss=[]
        for s in range(0,len(train_items),args.batch_size):
            ch=train_items[s:s+args.batch_size]; b=collate(ch,tok.pad_token_id)
            opt.zero_grad(set_to_none=True)
            logits=forward(model,b)
            target=b['target']
            loss=-(target*torch.log_softmax(logits,-1)).sum(-1).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(trainable,1.0); opt.step()
            ep_loss.append(float(loss.detach()))
        losses.append(float(np.mean(ep_loss)))
        print(f'fold={args.fold} epoch={ep+1} loss={losses[-1]:.6f}',flush=True)
    train_s=time.time()-train_start

    zcal,sym_cal=eval_laya(model,tok,cfg,calib,args.batch_size)
    temp=fit_temperature(zcal,calib.label.astype(int).to_numpy())
    ztest,sym_test=eval_laya(model,tok,cfg,test,args.batch_size)
    p_laya=1/(1+np.exp(-np.clip(ztest/temp,-40,40)))

    report={
        'protocol':'LayaETF head-only domain specialisation v1',
        'fold':args.fold,'train_year_max':args.fold-2,'calibration_year':cal_year,'test_year':args.fold,
        'n_train_pairs':int(len(train)),'n_train_items_with_swaps':int(len(train_items)),
        'n_calib_pairs':int(len(calib)),'n_test_pairs':int(len(test)),
        'model':'convaiinnovations/laya','encoder_frozen':True,'epochs':args.epochs,'batch_size':args.batch_size,
        'trainable_params':int(n_trainable),'model_load_s':load_s,'train_s':train_s,'epoch_losses':losses,
        'temperature':temp,'calibration_symmetry_error':sym_cal,'test_symmetry_error':sym_test,
        'metrics':{
            'laya_head_ft':metrics(yte,p_laya),
            'logistic':metrics(yte,p_log),
            'xgboost':metrics(yte,p_xgb),
            'hybrid24_top1':{'n':int(len(yte)),'accuracy':float(yte.mean())},
        }
    }
    la=report['metrics']['laya_head_ft']; lg=report['metrics']['logistic']; xg=report['metrics']['xgboost']
    report['signal_check']=bool(la['auc'] is not None and la['auc']>0.52 and la['auc']>max(lg['auc'],xg['auc']))
    out=Path(args.outdir)/f'fold_{args.fold}'; out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    pd.DataFrame({'case_id':test.case_id,'label':yte,'p_laya':p_laya,'p_logistic':p_log,'p_xgboost':p_xgb}).to_csv(out/'predictions.csv',index=False)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__': main()
