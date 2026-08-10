#!/usr/bin/env python3
"""Absolute error alongside the coefficient of determination across block
lengths, and a first comparison of a frozen model against periodic refitting.
"""
import numpy as np, pandas as pd, warnings, json
warnings.filterwarnings("ignore")
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA=str(ROOT / "data" / "UF_data.csv")
RES=str(ROOT / "results")
IN=['glu','mlss','air','fm','cn','hrt','srt']; TG=['tmp','flow','lv']; SEED=42
d=pd.read_csv(DATA); d.columns=[c.strip().lstrip('﻿') for c in d.columns]; n=len(d)
X=d[IN].values

def fit(m, y, p=None):
    sc=StandardScaler().fit(X[m])
    e=ExtraTreesRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)
    e.fit(sc.transform(X[m]), y[m])
    return e.predict(sc.transform(X[~m if p is None else p]))

# ---------------- Q1: error scale under the block sweep ----------------
print("Q1  ABSOLUTE ERROR VS R2 ACROSS BLOCK LENGTHS\n")
rows=[]
for tgt in TG:
    y=d[tgt].values; iqr=np.percentile(y,75)-np.percentile(y,25); sd=y.std()
    for bh in (1,24,168,720):
        r2s,maes=[],[]
        for s in range(3):
            rng=np.random.RandomState(11+s)
            b=np.arange(n)//bh; ids=np.unique(b); rng.shuffle(ids)
            tr=set(ids[:int(0.7*len(ids))]); m=np.array([x in tr for x in b])
            pr=fit(m,y); r2s.append(r2_score(y[~m],pr)); maes.append(mean_absolute_error(y[~m],pr))
        rows.append(dict(target=tgt,block_h=bh,R2=np.mean(r2s),MAE=np.mean(maes),
                         MAE_pct_of_IQR=100*np.mean(maes)/iqr, MAE_pct_of_SD=100*np.mean(maes)/sd))
    print(f"  {tgt.upper():5s} IQR={iqr:.4g}  SD={sd:.4g}")
E=pd.DataFrame(rows)
print(E.pivot(index='block_h',columns='target',values='R2').round(3).to_string(),"\n")
print("  MAE as % of the target's own IQR")
print(E.pivot(index='block_h',columns='target',values='MAE_pct_of_IQR').round(1).to_string())

# ---------------- Q2: frozen vs periodically refitted ----------------
print("\n\nQ2  FROZEN MODEL VS PERIODIC REFIT ON THE FINAL 30%\n")
cut=int(0.7*n); te=np.arange(n)>=cut
res=[]
for tgt in TG:
    y=d[tgt].values
    m=np.arange(n)<cut
    pr=fit(m,y)
    res.append(dict(target=tgt,mode='frozen at month 7',R2=r2_score(y[te],pr),
                    MAE=mean_absolute_error(y[te],pr)))
    for step,lab in [(720,'refit monthly'),(168,'refit weekly')]:
        P=np.full(n,np.nan)
        s=cut
        while s<n:
            e=min(s+step,n)
            tr=np.arange(n)<s
            sc=StandardScaler().fit(X[tr])
            mdl=ExtraTreesRegressor(n_estimators=200,random_state=SEED,n_jobs=-1)
            mdl.fit(sc.transform(X[tr]),y[tr])
            P[s:e]=mdl.predict(sc.transform(X[s:e]))
            s=e
        res.append(dict(target=tgt,mode=lab,R2=r2_score(y[te],P[te]),
                        MAE=mean_absolute_error(y[te],P[te])))
D=pd.DataFrame(res)
print(D.pivot(index='mode',columns='target',values='R2')
       .reindex(['frozen at month 7','refit monthly','refit weekly']).round(3).to_string())
print("\n  MAE")
print(D.pivot(index='mode',columns='target',values='MAE')
       .reindex(['frozen at month 7','refit monthly','refit weekly']).round(4).to_string())
E.to_csv(f"{RES}/H_error_scale.csv",index=False)
D.to_csv(f"{RES}/H_refit_vs_frozen.csv",index=False)
print("\nwritten")
