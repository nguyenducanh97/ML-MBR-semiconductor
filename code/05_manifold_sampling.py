#!/usr/bin/env python3
"""Local-covariance kernel resampling of the operating manifold.

A seed hour is drawn from the logged record, the local covariance of its k
nearest neighbours in standardised input space is formed, and the seed is
perturbed by a Gaussian increment scaled by that covariance. Candidates are
rejected if they leave the observed marginal range of any input or if their
nearest-neighbour distance to the record exceeds the 99th percentile of the
real-to-real nearest-neighbour distance. Feasibility, percentile windows and
per-variable recommendations are then computed on the accepted sample.
"""
import os, warnings, json
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import NearestNeighbors
from scipy.stats import gaussian_kde

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SEED = 42
DATA = str(ROOT / "data" / "UF_data.csv")
OUT  = str(ROOT / "results")
os.makedirs(OUT, exist_ok=True)
INPUTS  = ['glu','mlss','air','fm','cn','hrt','srt']
OUTPUTS = ['tmp','flow','lv']
TARGETS = {'tmp':(-0.09,-0.03), 'flow':(1.5,2.2), 'lv':(65.0,67.0)}
UNITS   = {'glu':'L/min','mlss':'mg/L','air':'m3/h','fm':'1/day','cn':'-','hrt':'h','srt':'day'}

df = pd.read_csv(DATA); df.columns=[c.strip().lstrip('﻿') for c in df.columns]
n = len(df); cut = int(0.7*n)
X = df[INPUTS].values
mu, sd = X.mean(0), X.std(0)
Xz = (X-mu)/sd

met = np.ones(n,bool)
for k,(lo,hi) in TARGETS.items():
    met &= (df[k].values>=lo)&(df[k].values<=hi)

# ---------------------------------------------------------------- models
print("Training Extra Trees on the full record ...")
models = {}
for t in OUTPUTS:
    m = ExtraTreesRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
    m.fit(Xz, df[t].values)
    models[t] = m
    print(f"  {t}: train R2 = {m.score(Xz, df[t].values):.4f}")

# ---------------------------------------------------------------- sampling
print("\nLocal-covariance kernel resampling...")
K, H = 30, 0.6
nn_k = NearestNeighbors(n_neighbors=K+1).fit(Xz)
_, nbr = nn_k.kneighbors(Xz)
covs = np.zeros((n,7,7))
for i in range(n):
    covs[i] = np.cov(Xz[nbr[i,1:]].T) + 1e-6*np.eye(7)
chol = np.zeros_like(covs)
for i in range(n):
    try: chol[i] = np.linalg.cholesky(covs[i])
    except np.linalg.LinAlgError:
        w,v = np.linalg.eigh(covs[i]); w = np.clip(w,1e-9,None)
        chol[i] = v@np.diag(np.sqrt(w))

nn1 = NearestNeighbors(n_neighbors=2).fit(Xz)
d_real,_ = nn1.kneighbors(Xz); d_real = d_real[:,1]
THR = float(np.percentile(d_real, 99))
print(f"  acceptance threshold (99th pct real NN distance) = {THR:.4f}")

nn_ref = NearestNeighbors(n_neighbors=1).fit(Xz)
lo_m, hi_m = X.min(0), X.max(0)
rng = np.random.RandomState(SEED)
TARGET_N = 500_000
acc = []
tries = 0
while sum(len(a) for a in acc) < TARGET_N and tries < 60:
    b = 200_000
    seeds = rng.randint(0, n, b)
    eps = rng.randn(b,7)
    cand_z = Xz[seeds] + H*np.einsum('bij,bj->bi', chol[seeds], eps)
    cand = cand_z*sd + mu
    ok = np.all((cand>=lo_m)&(cand<=hi_m), axis=1)
    cand, cand_z = cand[ok], cand_z[ok]
    d,_ = nn_ref.kneighbors(cand_z)
    ok2 = d.ravel() <= THR
    acc.append(cand[ok2])
    tries += 1
    tot = sum(len(a) for a in acc)
    print(f"  batch {tries}: accepted {ok2.sum():,} (total {tot:,})")
S = np.vstack(acc)[:TARGET_N]
print(f"  final sample: {len(S):,} candidate operating states")

Sz = (S-mu)/sd
d_s,_ = nn_ref.kneighbors(Sz)
print(f"  median NN distance of sample to real record: {np.median(d_s):.3f} "
      f"(real-to-real median {np.median(d_real):.3f})")

# ---------------------------------------------------------------- predict + feasible
pred = {t: models[t].predict(Sz) for t in OUTPUTS}
feas = np.ones(len(S), bool)
for t,(lo,hi) in TARGETS.items():
    feas &= (pred[t]>=lo)&(pred[t]<=hi)
print(f"\nFeasible points: {feas.sum():,} / {len(S):,} ({100*feas.mean():.1f}%)")

# ensemble prediction interval on the feasible set
pi = {}
for t in OUTPUTS:
    per_tree = np.stack([e.predict(Sz[feas][:20000]) for e in models[t].estimators_])
    pi[t] = float(1.96*per_tree.std(0).mean())
    print(f"  95% ensemble PI for {t}: +/- {pi[t]:.4f}")

F = pd.DataFrame(S, columns=INPUTS)
F['feasible'] = feas
for t in OUTPUTS: F['pred_'+t] = pred[t]
F[feas].sample(min(50000,int(feas.sum())), random_state=0).to_csv(
    f"{OUT}/E_feasible_points_manifold_sample.csv", index=False)

# ---------------------------------------------------------------- new Table 4
print("\n" + "="*100)
print("REVISED TABLE 4 - manifold-constrained feasible region, with real-data support")
print("="*100)
Ff = S[feas]
rows=[]
for j,f in enumerate(INPUTS):
    v = Ff[:,j]
    p5,p95 = np.percentile(v,[5,95])
    kde = gaussian_kde(v[np.random.RandomState(0).choice(len(v),min(20000,len(v)),replace=False)])
    grid = np.linspace(v.min(), v.max(), 400)
    dens = kde(grid)
    # preferred zone = shortest contiguous interval holding 40% of density mass
    cdf = np.cumsum(dens); cdf/=cdf[-1]
    best=(None,np.inf)
    for a in range(len(grid)):
        b = np.searchsorted(cdf, cdf[a]+0.40)
        if b>=len(grid): break
        w = grid[b]-grid[a]
        if w<best[1]: best=((grid[a],grid[b]),w)
    pz = best[0]
    real_in = (df[f].values>=pz[0])&(df[f].values<=pz[1])
    rows.append({'Parameter':f,'Unit':UNITS[f],
                 'Full range':f"{X[:,j].min():.4g}-{X[:,j].max():.4g}",
                 'Feasible P05-P95':f"{p5:.4g}-{p95:.4g}",
                 'Preferred zone':f"{pz[0]:.4g}-{pz[1]:.4g}",
                 'Real hours in zone':int(real_in.sum()),
                 'Pct of record':round(100*real_in.mean(),1),
                 'Pct met in zone':round(100*met[real_in].mean(),1) if real_in.sum() else np.nan,
                 'Pct met outside':round(100*met[~real_in].mean(),1)})
T4 = pd.DataFrame(rows)
T4.to_csv(f"{OUT}/E_revised_table4.csv", index=False)
print(T4.to_string(index=False))

# ---------------------------------------------------------------- joint support of new window
print("\n" + "="*100)
print("JOINT REAL-DATA SUPPORT OF THE REVISED WINDOW (cumulative)")
print("="*100)
zones = {r['Parameter']: tuple(float(x) for x in r['Preferred zone'].split('-')) if '-' in r['Preferred zone'][1:]
         else None for _,r in T4.iterrows()}
# robust parse
zones = {}
for _,r in T4.iterrows():
    s = r['Preferred zone']
    a,b = s.rsplit('-',1) if s.count('-')==1 else (s[:s.rfind('-')], s[s.rfind('-')+1:])
    zones[r['Parameter']] = (float(a), float(b))
order = ['srt','hrt','fm','mlss','air','cn','glu']
m = np.ones(n,bool); rows=[]
for k in order:
    lo,hi = zones[k]
    m = m & (df[k].values>=lo)&(df[k].values<=hi)
    rows.append({'up_to':k,'n_real_hours':int(m.sum()),'pct_record':round(100*m.mean(),2),
                 'pct_met':round(100*met[m].mean(),1) if m.sum() else np.nan})
js = pd.DataFrame(rows); js.to_csv(f"{OUT}/E_revised_joint_support.csv", index=False)
print(js.to_string(index=False))

# ---------------------------------------------------------------- out-of-time
print("\n" + "="*100)
print("OUT-OF-TIME CHECK OF THE REVISED PREFERRED ZONES (learned from full record,")
print("evaluated on the final 30% the models never saw)")
print("="*100)
hm = np.arange(n)>=cut
rows=[]
for k in INPUTS:
    lo,hi = zones[k]
    inw = (df[k].values>=lo)&(df[k].values<=hi)
    a, b = inw&hm, (~inw)&hm
    rows.append({'parameter':k,'zone':f"{lo:.4g}-{hi:.4g}",
                 'n_heldout_in':int(a.sum()),
                 'pct_met_in':round(100*met[a].mean(),1) if a.sum()>=30 else np.nan,
                 'n_heldout_out':int(b.sum()),
                 'pct_met_out':round(100*met[b].mean(),1) if b.sum()>=30 else np.nan,
                 'heldout_baseline':round(100*met[hm].mean(),1)})
oot = pd.DataFrame(rows); oot.to_csv(f"{OUT}/E_revised_out_of_time.csv", index=False)
print(oot.to_string(index=False))

json.dump({'threshold':THR,'n_sample':int(len(S)),'n_feasible':int(feas.sum()),
           'pct_feasible':float(100*feas.mean()),'PI95':pi,
           'median_NN_sample':float(np.median(d_s)),'median_NN_real':float(np.median(d_real))},
          open(f"{OUT}/E_manifold_meta.json","w"), indent=2)
print(f"\nOutputs -> {OUT}")
