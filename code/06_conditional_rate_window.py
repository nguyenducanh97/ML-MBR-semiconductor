#!/usr/bin/env python3
"""Conditional feasibility rate, p(all three targets met | x), profiled over
deciles of each input, together with the resulting per-variable bands and the
joint SRT-HRT rule.
"""
import os, warnings, json
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.neighbors import NearestNeighbors
from scipy.stats import fisher_exact

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SEED=42
DATA=str(ROOT / "data" / "UF_data.csv")
OUT=str(ROOT / "results")
INPUTS=['glu','mlss','air','fm','cn','hrt','srt']; OUTPUTS=['tmp','flow','lv']
TARGETS={'tmp':(-0.09,-0.03),'flow':(1.5,2.2),'lv':(65.0,67.0)}
UNITS={'glu':'L/min','mlss':'mg/L','air':'m3/h','fm':'1/day','cn':'-','hrt':'h','srt':'day'}

df=pd.read_csv(DATA); df.columns=[c.strip().lstrip('﻿') for c in df.columns]
n=len(df); cut=int(0.7*n); X=df[INPUTS].values
mu,sd=X.mean(0),X.std(0); Xz=(X-mu)/sd
met=np.ones(n,bool)
for k,(lo,hi) in TARGETS.items(): met&=(df[k].values>=lo)&(df[k].values<=hi)

S = pd.read_csv(f"{OUT}/E_feasible_points_manifold_sample.csv")
print("Regenerating full manifold sample (feasible + infeasible) for rate estimation...")
models={}
for t in OUTPUTS:
    m=ExtraTreesRegressor(n_estimators=100,random_state=SEED,n_jobs=-1); m.fit(Xz,df[t].values); models[t]=m

K,H=30,0.6
nn_k=NearestNeighbors(n_neighbors=K+1).fit(Xz); _,nbr=nn_k.kneighbors(Xz)
chol=np.zeros((n,7,7))
for i in range(n):
    C=np.cov(Xz[nbr[i,1:]].T)+1e-6*np.eye(7)
    try: chol[i]=np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        w,v=np.linalg.eigh(C); w=np.clip(w,1e-9,None); chol[i]=v@np.diag(np.sqrt(w))
nn1=NearestNeighbors(n_neighbors=2).fit(Xz); d_real,_=nn1.kneighbors(Xz); d_real=d_real[:,1]
THR=float(np.percentile(d_real,99)); nn_ref=NearestNeighbors(n_neighbors=1).fit(Xz)
lo_m,hi_m=X.min(0),X.max(0); rng=np.random.RandomState(SEED)
parts=[]
while sum(len(p) for p in parts)<400_000:
    b=200_000; s=rng.randint(0,n,b)
    cz=Xz[s]+H*np.einsum('bij,bj->bi',chol[s],rng.randn(b,7))
    c=cz*sd+mu
    ok=np.all((c>=lo_m)&(c<=hi_m),axis=1); c,cz=c[ok],cz[ok]
    d,_=nn_ref.kneighbors(cz); parts.append(c[d.ravel()<=THR])
Sall=np.vstack(parts)[:400_000]; Sz=(Sall-mu)/sd
pred={t:models[t].predict(Sz) for t in OUTPUTS}
feas=np.ones(len(Sall),bool)
for t,(lo,hi) in TARGETS.items(): feas&=(pred[t]>=lo)&(pred[t]<=hi)
print(f"  sample {len(Sall):,}, feasible {feas.sum():,} ({100*feas.mean():.1f}%)")

# ------------------------------------------------ 1. conditional rate profiles
print("\n"+"="*104)
print("1. CONDITIONAL FEASIBILITY RATE p(all targets met | bin) - MODEL vs REAL DATA")
print("="*104)
prof=[]
for j,f in enumerate(INPUTS):
    edges=np.percentile(X[:,j],np.linspace(0,100,11))
    edges=np.unique(edges)
    for a,b in zip(edges[:-1],edges[1:]):
        ms=(Sall[:,j]>=a)&(Sall[:,j]<b)
        mr=(X[:,j]>=a)&(X[:,j]<b)
        prof.append({'feature':f,'lo':round(float(a),4),'hi':round(float(b),4),
                     'n_model':int(ms.sum()),
                     'model_rate':round(100*feas[ms].mean(),1) if ms.sum()>200 else np.nan,
                     'n_real':int(mr.sum()),
                     'real_rate':round(100*met[mr].mean(),1) if mr.sum()>30 else np.nan})
P=pd.DataFrame(prof); P.to_csv(f"{OUT}/F_conditional_rate_profiles.csv",index=False)
for f in INPUTS:
    s=P[P.feature==f]
    print(f"\n{f} ({UNITS[f]})")
    print("   bin            model%  real%")
    for r in s.itertuples():
        print(f"   {r.lo:>9.4g}-{r.hi:<9.4g} {r.model_rate:>6} {r.real_rate:>6}")

# ------------------------------------------------ 2. preferred zone by rate
print("\n"+"="*104)
print("2. PREFERRED ZONE = contiguous band maximising the CONDITIONAL RATE")
print("="*104)
rows=[]
for j,f in enumerate(INPUTS):
    s=P[P.feature==f].reset_index(drop=True)
    best=None
    for a in range(len(s)):
        for b in range(a,min(a+4,len(s))):
            lo,hi=s.lo[a],s.hi[b]
            mr=(X[:,j]>=lo)&(X[:,j]<=hi)
            if mr.sum()<250: continue
            ms=(Sall[:,j]>=lo)&(Sall[:,j]<=hi)
            rr=met[mr].mean(); mm=feas[ms].mean() if ms.sum()>200 else np.nan
            score=rr
            if best is None or score>best['score']:
                best={'score':score,'lo':lo,'hi':hi,'real_rate':100*rr,'model_rate':100*mm,
                      'n_real':int(mr.sum())}
    out_mask=~((X[:,j]>=best['lo'])&(X[:,j]<=best['hi']))
    rows.append({'Parameter':f,'Unit':UNITS[f],
                 'Observed range':f"{X[:,j].min():.4g} - {X[:,j].max():.4g}",
                 'Recommended zone':f"{best['lo']:.4g} - {best['hi']:.4g}",
                 'Real hours in zone':best['n_real'],
                 'Attainment in zone %':round(best['real_rate'],1),
                 'Attainment outside %':round(100*met[out_mask].mean(),1),
                 'Model rate in zone %':round(best['model_rate'],1)})
T=pd.DataFrame(rows); T.to_csv(f"{OUT}/F_revised_table4_rate_based.csv",index=False)
print(T.to_string(index=False))

# ------------------------------------------------ 3. out-of-time
print("\n"+"="*104)
print("3. OUT-OF-TIME VALIDATION: zones derived on the FIRST 70%, tested on the FINAL 30%")
print("="*104)
tr=np.arange(n)<cut; te=~tr
rows=[]
for j,f in enumerate(INPUTS):
    s=P[P.feature==f].reset_index(drop=True)
    best=None
    for a in range(len(s)):
        for b in range(a,min(a+4,len(s))):
            lo,hi=s.lo[a],s.hi[b]
            mr=(X[:,j]>=lo)&(X[:,j]<=hi)&tr
            if mr.sum()<200: continue
            rr=met[mr].mean()
            if best is None or rr>best[0]: best=(rr,lo,hi,int(mr.sum()))
    if best is None: continue
    rr,lo,hi,ntr=best
    inw=(X[:,j]>=lo)&(X[:,j]<=hi)
    a_,b_=inw&te,(~inw)&te
    rows.append({'parameter':f,'zone_from_first70':f"{lo:.4g} - {hi:.4g}",
                 'train_attain%':round(100*rr,1),'n_train':ntr,
                 'n_heldout_in':int(a_.sum()),
                 'heldout_attain_in%':round(100*met[a_].mean(),1) if a_.sum()>=30 else np.nan,
                 'heldout_attain_out%':round(100*met[b_].mean(),1) if b_.sum()>=30 else np.nan,
                 'heldout_baseline%':round(100*met[te].mean(),1),
                 'lift_pp':round(100*(met[a_].mean()-met[te].mean()),1) if a_.sum()>=30 else np.nan})
O=pd.DataFrame(rows); O.to_csv(f"{OUT}/F_out_of_time_zones.csv",index=False)
print(O.to_string(index=False))

# ------------------------------------------------ 4. joint rule with real support
print("\n"+"="*104)
print("4. JOINT OPERATING RULE - learned on first 70%, tested on final 30%")
print("="*104)
clf=DecisionTreeClassifier(max_depth=3,min_samples_leaf=150,random_state=0)
clf.fit(X[tr],met[tr])
p_te=clf.predict(X[te]); y_te=met[te]
tbl=np.array([[int((y_te&(p_te==1)).sum()),int((~y_te&(p_te==1)).sum())],
              [int((y_te&(p_te==0)).sum()),int((~y_te&(p_te==0)).sum())]])
orr,pv=fisher_exact(tbl)
print(export_text(clf,feature_names=INPUTS))
print(f"train flagged {int(clf.predict(X[tr]).sum())}, precision {100*met[tr][clf.predict(X[tr])==1].mean():.1f}%")
print(f"held-out flagged {int(p_te.sum())}, precision {100*y_te[p_te==1].mean():.1f}%, "
      f"baseline {100*y_te.mean():.1f}%")
print(f"contingency {tbl.tolist()}  OR={orr:.2f}  Fisher p={pv:.3e}")
with open(f"{OUT}/F_joint_rule.txt","w") as fh:
    fh.write(export_text(clf,feature_names=INPUTS))
    fh.write(f"\nheld-out precision {100*y_te[p_te==1].mean():.1f}% vs baseline {100*y_te.mean():.1f}%\n")
    fh.write(f"contingency {tbl.tolist()} OR={orr:.3f} p={pv:.3e}\n")
json.dump({'heldout_precision':float(100*y_te[p_te==1].mean()),
           'heldout_baseline':float(100*y_te.mean()),'odds_ratio':float(orr),'p':float(pv),
           'n_flagged_heldout':int(p_te.sum())}, open(f"{OUT}/F_joint_rule.json","w"),indent=2)
print(f"\nOutputs -> {OUT}")
