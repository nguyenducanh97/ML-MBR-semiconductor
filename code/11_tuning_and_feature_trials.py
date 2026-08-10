#!/usr/bin/env python3
"""Whether blocked-CV accuracy responds to hyperparameter tuning with a blocked
inner loop or to rolling-history features derived from past input values.
"""
import numpy as np, pandas as pd, json, warnings, itertools, time
warnings.filterwarnings("ignore")
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = str(ROOT / "data" / "UF_data.csv")
RES  = str(ROOT / "results")
IN   = ['glu','mlss','air','fm','cn','hrt','srt']
TG   = ['tmp','flow','lv']
SEED = 42

d = pd.read_csv(DATA); d.columns=[c.strip().lstrip('﻿') for c in d.columns]
n = len(d)

# ---------- feature sets ----------
def raw(df):
    return df[IN].copy()

def history(df):
    """Recent operating history. Uses only PAST values of the seven inputs,
    which an operator has in hand at prediction time. No target is used."""
    X = df[IN].copy()
    for w in (24, 168):
        r = df[IN].rolling(w, min_periods=1)
        X = X.join(r.mean().add_suffix(f'_m{w}'))
        X = X.join(r.std().fillna(0).add_suffix(f'_s{w}'))
    for L in (24, 168):
        X = X.join((df[IN] - df[IN].shift(L)).fillna(0).add_suffix(f'_d{L}'))
    return X

FEATS = {'raw': raw, 'history': history}

# ---------- blocked allocation ----------
def blocks(block_h, rng):
    b = np.arange(n) // block_h
    ids = np.unique(b); rng.shuffle(ids)
    tr = set(ids[:int(0.7*len(ids))])
    return np.array([x in tr for x in b])

def fit_eval(X, y, m, params):
    sc = StandardScaler().fit(X[m])
    e = ExtraTreesRegressor(random_state=SEED, n_jobs=-1, **params)
    e.fit(sc.transform(X[m]), y[m])
    return r2_score(y[~m], e.predict(sc.transform(X[~m])))

BASE = dict(n_estimators=100)
GRID = [dict(n_estimators=150, max_features=f, min_samples_leaf=l, max_depth=dp)
        for f, l, dp in itertools.product([0.5, 1.0], [1, 10], [None])]

t0 = time.time(); out = []
for tgt in TG:
    y = d[tgt].values
    for fname, fn in FEATS.items():
        X = fn(d).values
        # ---- inner tuning on the TRAINING portion only, blocked inner folds
        rng = np.random.RandomState(7)
        inner = [blocks(24, rng) for _ in range(3)]
        best, bp = -9, BASE
        for p in GRID:
            s = np.mean([fit_eval(X, y, m, p) for m in inner])
            if s > best: best, bp = s, p
        for label, params in [('default', BASE), ('tuned', bp)]:
            for bh in (1, 24, 168, 720):
                rng2 = np.random.RandomState(11)
                sc = [fit_eval(X, y, blocks(bh, rng2), params) for _ in range(3)]
                out.append(dict(target=tgt, features=fname, params=label, design=f'blocked {bh} h',
                                R2=np.mean(sc), sd=np.std(sc), cfg=str(params)))
            m = np.arange(n) < int(0.7*n)
            out.append(dict(target=tgt, features=fname, params=label, design='chronological',
                            R2=fit_eval(X, y, m, params), sd=0, cfg=str(params)))
        print(f"  {tgt:5s} {fname:8s} best={bp}  ({time.time()-t0:.0f}s)")

R = pd.DataFrame(out)
R.to_csv(f"{RES}/H_enhancement_grid.csv", index=False)
print("\n=== TEST R2 ===")
for tgt in TG:
    print(f"\n{tgt.upper()}")
    p = R[R.target==tgt].pivot_table(index='design', columns=['features','params'], values='R2')
    print(p.reindex(['blocked 1 h','blocked 24 h','blocked 168 h','blocked 720 h','chronological']).round(3).to_string())
