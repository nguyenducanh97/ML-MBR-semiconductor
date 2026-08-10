#!/usr/bin/env python3
"""Accuracy on the held-out final 30% of the record as a function of how often
the model is refitted on the data logged so far. Every prediction is made
strictly ahead of the data used to produce it.
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
d=pd.read_csv(DATA); d.columns=[c.strip().lstrip('﻿') for c in d.columns]
n=len(d); X=d[IN].values; cut=int(0.7*n); te=np.arange(n)>=cut
IQR={t:np.percentile(d[t],75)-np.percentile(d[t],25) for t in TG}

def et():
    return ExtraTreesRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)

def train_predict(tr, pred_idx, y):
    """Exactly the protocol used for the published chronological result:
    inputs and target are both z-scored on the training fold, and the
    prediction is mapped back to physical units."""
    Xtr = X[tr]; mux, sdx = Xtr.mean(0), Xtr.std(0); sdx[sdx == 0] = 1.0
    ytr = y[tr]; muy, sdy = ytr.mean(), ytr.std() or 1.0
    m = et(); m.fit((Xtr - mux) / sdx, (ytr - muy) / sdy)
    return m.predict((X[pred_idx] - mux) / sdx) * sdy + muy

def rolling(y, step):
    """Refit on every hour logged before the current window, predict the window."""
    P = np.full(n, np.nan); s = cut
    while s < n:
        e = min(s + step, n)
        P[s:e] = train_predict(np.arange(n) < s, np.arange(s, e), y)
        s = e
    return P

rows=[]
for tgt in TG:
    y=d[tgt].values
    P0=train_predict(np.arange(n)<cut, te, y)
    rows.append(dict(target=tgt, mode='frozen', interval_h=np.nan,
                     R2=r2_score(y[te],P0), MAE=mean_absolute_error(y[te],P0)))
    for step,lab in [(720,'monthly'),(336,'fortnightly'),(168,'weekly'),(24,'daily')]:
        P=rolling(y,step)
        rows.append(dict(target=tgt, mode=lab, interval_h=step,
                         R2=r2_score(y[te],P[te]), MAE=mean_absolute_error(y[te],P[te])))
    print(" ",tgt,"done")

R=pd.DataFrame(rows)
R['MAE_pct_IQR']=[100*r.MAE/IQR[r.target] for r in R.itertuples()]
R.to_csv(f"{RES}/H_refit_schedule.csv", index=False)
order=['frozen','monthly','fortnightly','weekly','daily']
print("\nTEST R2 ON MONTHS 8-11")
print(R.pivot(index='mode',columns='target',values='R2').reindex(order).round(3).to_string())
print("\nMAE")
print(R.pivot(index='mode',columns='target',values='MAE').reindex(order).round(4).to_string())
print("\nMAE AS % OF THE TARGET'S IQR")
print(R.pivot(index='mode',columns='target',values='MAE_pct_IQR').reindex(order).round(1).to_string())
