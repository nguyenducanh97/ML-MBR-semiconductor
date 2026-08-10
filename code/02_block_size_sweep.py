#!/usr/bin/env python3
"""Autocorrelation of the record, test accuracy as a function of the contiguous
block length used for splitting, nearest-neighbour distances between the
training and test folds, and accuracy stratified by covariate coverage.
"""
import os, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.base import clone
from sklearn.neighbors import NearestNeighbors

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RANDOM_SEED = 42
DATA = str(ROOT / "data" / "UF_data.csv")
OUT  = str(ROOT / "results")
os.makedirs(OUT, exist_ok=True)

INPUTS  = ['glu', 'mlss', 'air', 'fm', 'cn', 'hrt', 'srt']
OUTPUTS = ['tmp', 'flow', 'lv']

df = pd.read_csv(DATA); df.columns = [c.strip().lstrip('﻿') for c in df.columns]
n = len(df)
X_raw = df[INPUTS].values

ET = ExtraTreesRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)

def run(tr, te, tgt, proto=ET):
    mux, sdx = X_raw[tr].mean(0), X_raw[tr].std(0); sdx[sdx == 0] = 1
    Xtr, Xte = (X_raw[tr]-mux)/sdx, (X_raw[te]-mux)/sdx
    y = df[tgt].values
    muy, sdy = y[tr].mean(), y[tr].std() or 1
    m = clone(proto); m.fit(Xtr, (y[tr]-muy)/sdy)
    p = m.predict(Xte)*sdy + muy
    return (r2_score(y[te], p), float(np.sqrt(mean_squared_error(y[te], p))),
            mean_absolute_error(y[te], p), p)

# ============================================================ 1. error magnitudes
print("=" * 78)
print("1. ABSOLUTE ERROR: is the negative R2 real, or a low-variance artefact?")
print("=" * 78)
rng = np.random.RandomState(RANDOM_SEED)
perm = rng.permutation(n); cut = int(0.7*n)
schemes = {'random_70_30': (perm[:cut], perm[cut:]),
           'chronological_70_30': (np.arange(cut), np.arange(cut, n))}
rows = []
for tgt in OUTPUTS:
    for tag, (tr, te) in schemes.items():
        r2, rmse, mae, _ = run(tr, te, tgt)
        rows.append({'target': tgt, 'split': tag, 'R2': r2, 'RMSE': rmse, 'MAE': mae,
                     'test_SD': df[tgt].values[te].std(),
                     'train_SD': df[tgt].values[tr].std(),
                     'test_mean': df[tgt].values[te].mean(),
                     'train_mean': df[tgt].values[tr].mean()})
err = pd.DataFrame(rows)
err.to_csv(f"{OUT}/1_error_magnitudes.csv", index=False)
print(err.round(4).to_string(index=False))

# ============================================================ 2. block-size sweep
print("\n" + "=" * 78)
print("2. BLOCK-SIZE SWEEP  (envelope preserved, temporal neighbours removed)")
print("=" * 78)
block_sizes = [1, 6, 24, 72, 168, 336, 720]
N_REP = 5
sweep = []
for bs in block_sizes:
    nb = int(np.ceil(n / bs))
    blocks = [np.arange(i*bs, min((i+1)*bs, n)) for i in range(nb)]
    for rep in range(N_REP):
        r = np.random.RandomState(1000 + rep)
        order = r.permutation(nb)
        n_tr_b = int(0.7*nb)
        tr = np.concatenate([blocks[i] for i in order[:n_tr_b]])
        te = np.concatenate([blocks[i] for i in order[n_tr_b:]])
        for tgt in OUTPUTS:
            r2, rmse, mae, _ = run(tr, te, tgt)
            sweep.append({'block_h': bs, 'rep': rep, 'target': tgt,
                          'R2': r2, 'RMSE': rmse, 'MAE': mae})
    print(f"  block = {bs:4d} h done")
sw = pd.DataFrame(sweep)
sw.to_csv(f"{OUT}/2_block_size_sweep_raw.csv", index=False)
agg = sw.groupby(['target', 'block_h']).agg(
    R2_mean=('R2', 'mean'), R2_sd=('R2', 'std'),
    RMSE_mean=('RMSE', 'mean'), MAE_mean=('MAE', 'mean')).reset_index()
agg.to_csv(f"{OUT}/2_block_size_sweep_summary.csv", index=False)
print()
for tgt in OUTPUTS:
    print(f"--- {tgt} ---")
    print(agg[agg.target == tgt].round(4).to_string(index=False))

# ============================================================ 3. NN distance
print("\n" + "=" * 78)
print("3. NEAREST-NEIGHBOUR DISTANCE from each test point to the training set")
print("=" * 78)
mux, sdx = X_raw.mean(0), X_raw.std(0)
Xz = (X_raw - mux)/sdx
nn_rows = []
for tag, (tr, te) in schemes.items():
    nn = NearestNeighbors(n_neighbors=1).fit(Xz[tr])
    d, _ = nn.kneighbors(Xz[te])
    nn_rows.append({'split': tag, 'median_NN_dist': float(np.median(d)),
                    'p90_NN_dist': float(np.percentile(d, 90)),
                    'mean_NN_dist': float(d.mean())})
# block sweep NN distances
for bs in block_sizes:
    nb = int(np.ceil(n/bs))
    blocks = [np.arange(i*bs, min((i+1)*bs, n)) for i in range(nb)]
    r = np.random.RandomState(1000); order = r.permutation(nb); n_tr_b = int(0.7*nb)
    tr = np.concatenate([blocks[i] for i in order[:n_tr_b]])
    te = np.concatenate([blocks[i] for i in order[n_tr_b:]])
    nn = NearestNeighbors(n_neighbors=1).fit(Xz[tr]); d, _ = nn.kneighbors(Xz[te])
    nn_rows.append({'split': f'block_{bs}h', 'median_NN_dist': float(np.median(d)),
                    'p90_NN_dist': float(np.percentile(d, 90)), 'mean_NN_dist': float(d.mean())})
nnd = pd.DataFrame(nn_rows)
nnd.to_csv(f"{OUT}/3_nn_distances.csv", index=False)
print(nnd.round(4).to_string(index=False))

# ============================================================ 4. covariate coverage
print("\n" + "=" * 78)
print("4. COVARIATE SHIFT on the forward split: is the last 30% inside the")
print("   operating envelope the first 70% ever visited?")
print("=" * 78)
tr, te = np.arange(cut), np.arange(cut, n)
lo = np.percentile(X_raw[tr], 1, axis=0)
hi = np.percentile(X_raw[tr], 99, axis=0)
inside_mat = (X_raw[te] >= lo) & (X_raw[te] <= hi)
cov = pd.DataFrame({'feature': INPUTS,
                    'train_p1': lo.round(3), 'train_p99': hi.round(3),
                    'test_min': X_raw[te].min(0).round(3),
                    'test_max': X_raw[te].max(0).round(3),
                    'pct_test_inside': (inside_mat.mean(0)*100).round(1)})
print(cov.to_string(index=False))
cov.to_csv(f"{OUT}/4_covariate_coverage.csv", index=False)

all_inside = inside_mat.all(axis=1)
print(f"\nTest points inside the training envelope on ALL 7 features: "
      f"{all_inside.sum()} / {len(te)} ({100*all_inside.mean():.1f}%)")

strat = []
for tgt in OUTPUTS:
    r2, rmse, mae, pred = run(tr, te, tgt)
    y = df[tgt].values[te]
    for label, mask in [('inside_envelope', all_inside), ('outside_envelope', ~all_inside)]:
        if mask.sum() < 30:
            strat.append({'target': tgt, 'subset': label, 'n': int(mask.sum()),
                          'R2': np.nan, 'RMSE': np.nan, 'MAE': np.nan}); continue
        strat.append({'target': tgt, 'subset': label, 'n': int(mask.sum()),
                      'R2': r2_score(y[mask], pred[mask]),
                      'RMSE': float(np.sqrt(mean_squared_error(y[mask], pred[mask]))),
                      'MAE': mean_absolute_error(y[mask], pred[mask])})
st = pd.DataFrame(strat)
st.to_csv(f"{OUT}/4_stratified_by_envelope.csv", index=False)
print("\nForward-split performance stratified by envelope membership:")
print(st.round(4).to_string(index=False))
print(f"\nOutputs -> {OUT}")
