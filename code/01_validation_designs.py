#!/usr/bin/env python3
"""Test-set performance of all 16 models under random, chronological and
purged chronological partitions, and under blocked cross-validation.
"""
import os, json, warnings, time
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, AdaBoostRegressor,
                              HistGradientBoostingRegressor, BaggingRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.base import clone
import xgboost as xgb
import lightgbm as lgb

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RANDOM_SEED = 42
DATA = str(ROOT / "data" / "UF_data.csv")
OUT  = str(ROOT / "results")
os.makedirs(OUT, exist_ok=True)

INPUTS  = ['glu', 'mlss', 'air', 'fm', 'cn', 'hrt', 'srt']
OUTPUTS = ['tmp', 'flow', 'lv']

MODELS = {
    'Linear Regression':      LinearRegression(),
    'Ridge Regression':       Ridge(alpha=1.0, random_state=RANDOM_SEED),
    'Lasso Regression':       Lasso(alpha=0.1, random_state=RANDOM_SEED, max_iter=10000),
    'ElasticNet':             ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=RANDOM_SEED, max_iter=10000),
    'Decision Tree':          DecisionTreeRegressor(random_state=RANDOM_SEED, max_depth=10),
    'Random Forest':          RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1),
    'Extra Trees':            ExtraTreesRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1),
    'Gradient Boosting':      GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_SEED),
    'AdaBoost':               AdaBoostRegressor(n_estimators=100, random_state=RANDOM_SEED),
    'Hist Gradient Boosting': HistGradientBoostingRegressor(random_state=RANDOM_SEED),
    'KNN':                    KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
    'SVR':                    SVR(kernel='rbf', C=1.0),
    'Bagging':                BaggingRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1),
    'MLP Neural Network':     MLPRegressor(hidden_layer_sizes=(100, 50), random_state=RANDOM_SEED,
                                           max_iter=1000, early_stopping=True),
    'XGBoost':                xgb.XGBRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1),
    'LightGBM':               lgb.LGBMRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1, verbose=-1),
}

df = pd.read_csv(DATA)
df.columns = [c.strip().lstrip('﻿') for c in df.columns]
n = len(df)
print(f"Loaded {n} hourly observations, columns: {list(df.columns)}")

# ---------------------------------------------------------------- A. autocorrelation
def acf(x, nlags):
    x = np.asarray(x, float); x = x - x.mean()
    d = np.dot(x, x)
    return np.array([1.0] + [np.dot(x[:-k], x[k:]) / d for k in range(1, nlags + 1)])

rows = []
for c in INPUTS + OUTPUTS:
    a = acf(df[c].values, 200)
    # decorrelation lag: first lag where acf drops below 1/e
    thr = 1.0 / np.e
    below = np.where(a < thr)[0]
    tau = int(below[0]) if len(below) else 200
    # effective sample size (Bartlett)
    ess = n / (1 + 2 * np.sum(a[1:tau + 1])) if tau > 0 else n
    rows.append({'variable': c, 'acf_lag1': round(a[1], 4), 'acf_lag6': round(a[6], 4),
                 'acf_lag24': round(a[24], 4), 'acf_lag168': round(a[168], 4),
                 'decorrelation_lag_h': tau, 'effective_N': int(max(ess, 1))})
acf_df = pd.DataFrame(rows)
acf_df.to_csv(f"{OUT}/A_autocorrelation.csv", index=False)
print("\n=== A. AUTOCORRELATION ===")
print(acf_df.to_string(index=False))

# ---------------------------------------------------------------- helpers
def zscore_fit(tr):
    mu, sd = tr.mean(0), tr.std(0)
    sd[sd == 0] = 1.0
    return mu, sd

def evaluate(tr_idx, te_idx, tag):
    Xtr_r, Xte_r = df[INPUTS].values[tr_idx], df[INPUTS].values[te_idx]
    mux, sdx = zscore_fit(Xtr_r)
    Xtr, Xte = (Xtr_r - mux) / sdx, (Xte_r - mux) / sdx
    res = []
    for tgt in OUTPUTS:
        ytr_r, yte_r = df[tgt].values[tr_idx], df[tgt].values[te_idx]
        muy, sdy = ytr_r.mean(), ytr_r.std() or 1.0
        ytr = (ytr_r - muy) / sdy
        for name, proto in MODELS.items():
            m = clone(proto)
            t0 = time.time()
            m.fit(Xtr, ytr)
            pred = m.predict(Xte) * sdy + muy
            res.append({'split': tag, 'target': tgt, 'model': name,
                        'R2':   r2_score(yte_r, pred),
                        'RMSE': float(np.sqrt(mean_squared_error(yte_r, pred))),
                        'MAE':  mean_absolute_error(yte_r, pred),
                        'fit_s': round(time.time() - t0, 3)})
    return res

all_res = []

# ---------------------------------------------------------------- B. random split
rng = np.random.RandomState(RANDOM_SEED)
perm = rng.permutation(n)
cut = int(0.7 * n)
all_res += evaluate(perm[:cut], perm[cut:], 'random_70_30')
print("\nB. random split done")

# ---------------------------------------------------------------- C. chronological
all_res += evaluate(np.arange(0, cut), np.arange(cut, n), 'chronological_70_30')
print("C. chronological split done")

# ---------------------------------------------------------------- D. purged chronological
GAP = 168  # one week
all_res += evaluate(np.arange(0, cut - GAP), np.arange(cut, n), f'purged_gap{GAP}h')
print("D. purged split done")

res_df = pd.DataFrame(all_res)
res_df.to_csv(f"{OUT}/B_D_split_comparison_raw.csv", index=False)

# pivot: R2 by model x target x split
piv = res_df.pivot_table(index='model', columns=['target', 'split'], values='R2')
piv.to_csv(f"{OUT}/B_D_split_comparison_R2.csv")
print("\n=== B-D. TEST R2 BY SPLIT STRATEGY ===")
for tgt in OUTPUTS:
    sub = res_df[res_df.target == tgt].pivot(index='model', columns='split', values='R2')
    sub = sub[['random_70_30', 'chronological_70_30', f'purged_gap{GAP}h']]
    sub['drop_random_to_chrono'] = sub['random_70_30'] - sub['chronological_70_30']
    sub = sub.sort_values('random_70_30', ascending=False)
    print(f"\n--- {tgt} ---")
    print(sub.round(4).to_string())

# ---------------------------------------------------------------- E. blocked TSCV
print("\n=== E. BLOCKED (EXPANDING-WINDOW) TIME-SERIES CV, 5 folds ===")
tscv_rows = []
n_folds = 5
fold_size = n // (n_folds + 1)
for tgt in OUTPUTS:
    for name, proto in MODELS.items():
        scores = []
        for k in range(1, n_folds + 1):
            tr = np.arange(0, k * fold_size)
            te = np.arange(k * fold_size, min((k + 1) * fold_size, n))
            if len(te) < 50:
                continue
            Xtr_r, Xte_r = df[INPUTS].values[tr], df[INPUTS].values[te]
            mux, sdx = zscore_fit(Xtr_r)
            Xtr, Xte = (Xtr_r - mux) / sdx, (Xte_r - mux) / sdx
            ytr_r, yte_r = df[tgt].values[tr], df[tgt].values[te]
            muy, sdy = ytr_r.mean(), ytr_r.std() or 1.0
            m = clone(proto); m.fit(Xtr, (ytr_r - muy) / sdy)
            scores.append(r2_score(yte_r, m.predict(Xte) * sdy + muy))
        tscv_rows.append({'target': tgt, 'model': name,
                          'blockedCV_R2_mean': np.mean(scores),
                          'blockedCV_R2_std': np.std(scores),
                          'folds': json.dumps([round(s, 4) for s in scores])})
tscv_df = pd.DataFrame(tscv_rows)
tscv_df.to_csv(f"{OUT}/E_blocked_tscv.csv", index=False)
for tgt in OUTPUTS:
    print(f"\n--- {tgt} ---")
    print(tscv_df[tscv_df.target == tgt].sort_values('blockedCV_R2_mean', ascending=False)
          [['model', 'blockedCV_R2_mean', 'blockedCV_R2_std']].round(4).to_string(index=False))

# ---------------------------------------------------------------- F. summary
summary = []
for tgt in OUTPUTS:
    r = res_df[(res_df.target == tgt)].set_index(['model', 'split'])['R2']
    for name in MODELS:
        summary.append({
            'target': tgt, 'model': name,
            'random': r.get((name, 'random_70_30'), np.nan),
            'chronological': r.get((name, 'chronological_70_30'), np.nan),
            'purged': r.get((name, f'purged_gap{GAP}h'), np.nan),
            'blockedCV': tscv_df[(tscv_df.target == tgt) & (tscv_df.model == name)]['blockedCV_R2_mean'].iloc[0],
        })
sm = pd.DataFrame(summary)
sm['optimism_random_minus_chrono'] = sm['random'] - sm['chronological']
sm.to_csv(f"{OUT}/F_summary_all_splits.csv", index=False)
print("\n=== F. SUMMARY WRITTEN ===")
print(sm[sm.model == 'Extra Trees'].round(4).to_string(index=False))
print(f"\nAll outputs -> {OUT}")
