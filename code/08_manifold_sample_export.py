#!/usr/bin/env python3
"""Export the full manifold sample, feasible and infeasible points alike, and
the per-model blocked cross-validation table.
"""
import os, warnings, json
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, AdaBoostRegressor,
                              HistGradientBoostingRegressor, BaggingRegressor)
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.base import clone
import xgboost as xgb, lightgbm as lgb

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SEED = 42
DATA = str(ROOT / "data" / "UF_data.csv")
OUT = str(ROOT / "results")
INPUTS = ['glu', 'mlss', 'air', 'fm', 'cn', 'hrt', 'srt']
OUTPUTS = ['tmp', 'flow', 'lv']
TARGETS = {'tmp': (-0.09, -0.03), 'flow': (1.5, 2.2), 'lv': (65.0, 67.0)}

df = pd.read_csv(DATA); df.columns = [c.strip().lstrip('﻿') for c in df.columns]
n = len(df); cut = int(0.7 * n)
X = df[INPUTS].values
mu, sd = X.mean(0), X.std(0); Xz = (X - mu) / sd

# ---------------------------------------------------------------- 1. full sample
print("Regenerating the manifold sample and retaining ALL candidates ...")
models = {}
for t in OUTPUTS:
    m = ExtraTreesRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
    m.fit(Xz, df[t].values); models[t] = m

K, H = 30, 0.6
nn_k = NearestNeighbors(n_neighbors=K + 1).fit(Xz); _, nbr = nn_k.kneighbors(Xz)
chol = np.zeros((n, 7, 7))
for i in range(n):
    C = np.cov(Xz[nbr[i, 1:]].T) + 1e-6 * np.eye(7)
    try:
        chol[i] = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        w, v = np.linalg.eigh(C); w = np.clip(w, 1e-9, None); chol[i] = v @ np.diag(np.sqrt(w))
nn1 = NearestNeighbors(n_neighbors=2).fit(Xz); d_real, _ = nn1.kneighbors(Xz); d_real = d_real[:, 1]
THR = float(np.percentile(d_real, 99))
nn_ref = NearestNeighbors(n_neighbors=1).fit(Xz)
lo_m, hi_m = X.min(0), X.max(0)
rng = np.random.RandomState(SEED)

parts = []
while sum(len(p) for p in parts) < 500_000:
    b = 200_000
    s = rng.randint(0, n, b)
    cz = Xz[s] + H * np.einsum('bij,bj->bi', chol[s], rng.randn(b, 7))
    c = cz * sd + mu
    ok = np.all((c >= lo_m) & (c <= hi_m), axis=1)
    c, cz = c[ok], cz[ok]
    d, _ = nn_ref.kneighbors(cz)
    parts.append(c[d.ravel() <= THR])
S = np.vstack(parts)[:500_000]
Sz = (S - mu) / sd
pred = {t: models[t].predict(Sz) for t in OUTPUTS}
feas = np.ones(len(S), bool)
for t, (lo, hi) in TARGETS.items():
    feas &= (pred[t] >= lo) & (pred[t] <= hi)
print(f"  full sample {len(S):,}; feasible {feas.sum():,} ({100*feas.mean():.1f}%)")

full = pd.DataFrame(S, columns=INPUTS)
for t in OUTPUTS: full['pred_' + t] = pred[t]
full['feasible'] = feas
idx = np.random.RandomState(1).choice(len(full), 150_000, replace=False)
full.iloc[np.sort(idx)].to_csv(f"{OUT}/E_manifold_sample_FULL.csv", index=False)
full[feas].sample(min(60000, int(feas.sum())), random_state=0).to_csv(
    f"{OUT}/E_feasible_points_manifold_sample.csv", index=False)
print("  saved full and feasible samples")

# ---------------------------------------------------------------- 2. S10 table
print("\nBuilding per-model blocked-validation table ...")
MODELS = {
    'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=SEED, n_jobs=-1),
    'Bagging': BaggingRegressor(n_estimators=100, random_state=SEED, n_jobs=-1),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=SEED, n_jobs=-1),
    'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=SEED, n_jobs=-1, verbose=-1),
    'Hist Grad. Boost.': HistGradientBoostingRegressor(random_state=SEED),
    'KNN': KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
    'MLP Neural Net.': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=SEED,
                                    max_iter=1000, early_stopping=True),
    'Decision Tree': DecisionTreeRegressor(random_state=SEED, max_depth=10),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=SEED),
    'SVR': SVR(kernel='rbf', C=1.0),
    'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=SEED),
    'Lin Regression': LinearRegression(),
    'Rid Regression': Ridge(alpha=1.0, random_state=SEED),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=SEED, max_iter=10000),
    'Las Regression': Lasso(alpha=0.1, random_state=SEED, max_iter=10000),
}

def blocked_idx(bs, seed):
    nb = int(np.ceil(n / bs))
    B = [np.arange(i * bs, min((i + 1) * bs, n)) for i in range(nb)]
    o = np.random.RandomState(seed).permutation(nb); k = int(0.7 * nb)
    return (np.sort(np.concatenate([B[i] for i in o[:k]])),
            np.sort(np.concatenate([B[i] for i in o[k:]])))

def fit_eval(proto, tr, te, tgt):
    m_, s_ = X[tr].mean(0), X[tr].std(0); s_[s_ == 0] = 1
    Xtr, Xte = (X[tr] - m_) / s_, (X[te] - m_) / s_
    y = df[tgt].values; my, sy = y[tr].mean(), y[tr].std() or 1
    mdl = clone(proto); mdl.fit(Xtr, (y[tr] - my) / sy)
    return r2_score(y[te], mdl.predict(Xte) * sy + my)

rs = np.random.RandomState(SEED); perm = rs.permutation(n)
rnd = (np.sort(perm[:cut]), np.sort(perm[cut:]))
rows = []
for name, proto in MODELS.items():
    for tgt in OUTPUTS:
        r_rand = fit_eval(proto, *rnd, tgt)
        r24 = np.mean([fit_eval(proto, *blocked_idx(24, 1000 + k), tgt) for k in range(3)])
        r168 = np.mean([fit_eval(proto, *blocked_idx(168, 1000 + k), tgt) for k in range(3)])
        rows.append({'model': name, 'target': tgt, 'random': r_rand, 'b24': r24, 'b168': r168})
    print(f"  {name} done")
pd.DataFrame(rows).to_csv(f"{OUT}/S10_blocked_per_model.csv", index=False)
print("\nS10 written. Extra Trees rank check:")
bm = pd.DataFrame(rows)
for t in OUTPUTS:
    s = bm[bm.target == t]
    for col in ['random', 'b24', 'b168']:
        top = s.sort_values(col, ascending=False).iloc[0]
        print(f"  {t:4s} {col:6s} best = {top.model} ({top[col]:.3f})")
