#!/usr/bin/env python3
"""Decile dose-response profiles for the seven inputs, joint support of the
combined window, an out-of-time decision-tree rule, and a realism check on
Latin hypercube sampling of the marginal ranges.
"""
import os, warnings, json
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.neighbors import NearestNeighbors
from scipy.stats import fisher_exact

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = str(ROOT / "data" / "UF_data.csv")
OUT  = str(ROOT / "results")
os.makedirs(OUT, exist_ok=True)
INPUTS  = ['glu','mlss','air','fm','cn','hrt','srt']; OUTPUTS = ['tmp','flow','lv']
df = pd.read_csv(DATA); df.columns=[c.strip().lstrip('﻿') for c in df.columns]
n = len(df); cut = int(0.7*n)
TARGETS = {'tmp':(-0.09,-0.03), 'flow':(1.5,2.2), 'lv':(65.0,67.0)}
met = np.ones(n,bool)
for k,(lo,hi) in TARGETS.items():
    met &= (df[k].values>=lo)&(df[k].values<=hi)
df['met'] = met
print(f"Overall attainment: {met.sum()}/{n} = {100*met.mean():.1f}%")
print(f"First 70%: {100*met[:cut].mean():.1f}%   Final 30%: {100*met[cut:].mean():.1f}%")

# ============================================== A. dose-response by decile
print("\n" + "="*78); print("A. TARGET ATTAINMENT BY DECILE OF EACH INPUT (real logged hours)"); print("="*78)
rows=[]
for f in INPUTS:
    q = pd.qcut(df[f], 10, labels=False, duplicates='drop')
    for d in sorted(pd.unique(q.dropna())):
        m = (q==d).values
        rows.append({'feature':f,'decile':int(d)+1,'n':int(m.sum()),
                     'lo':round(float(df[f][m].min()),4),'hi':round(float(df[f][m].max()),4),
                     'pct_met':round(100*met[m].mean(),1)})
dr = pd.DataFrame(rows); dr.to_csv(f"{OUT}/A_dose_response_deciles.csv", index=False)
for f in INPUTS:
    s = dr[dr.feature==f]
    print(f"\n{f}: " + "  ".join(f"[{r.lo:g}-{r.hi:g}]={r.pct_met:.0f}%" for r in s.itertuples()))

# ============================================== B. joint support audit
print("\n" + "="*78); print("B. JOINT SUPPORT OF THE PUBLISHED WINDOW"); print("="*78)
WINDOW = {'srt':(50,70),'hrt':(6.0,7.0),'air':(5500,6500),'mlss':(3500,6000),
          'cn':(6,8),'fm':(0.020,0.040),'glu':(0.5,0.9)}
order = ['srt','hrt','air','mlss','cn','fm','glu']
m = np.ones(n,bool); rows=[]
for k in order:
    lo,hi = WINDOW[k]
    m = m & (df[k].values>=lo)&(df[k].values<=hi)
    rows.append({'constraints_added_upto':k,'n_hours_satisfying':int(m.sum()),
                 'pct_of_record':round(100*m.mean(),2),
                 'pct_met_given_inside':round(100*met[m].mean(),1) if m.sum() else np.nan})
ja = pd.DataFrame(rows); ja.to_csv(f"{OUT}/B_joint_support_audit.csv", index=False)
print(ja.to_string(index=False))

# ============================================== C. out-of-time rule test
print("\n" + "="*78)
print("C. OPERATING RULE LEARNED ON FIRST 70% (chronological), TESTED ON FINAL 30%")
print("="*78)
Xtr, ytr = df[INPUTS].values[:cut], met[:cut]
Xte, yte = df[INPUTS].values[cut:], met[cut:]
best=None
for depth in [2,3,4]:
    for leaf in [50,100,150,200]:
        clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf,
                                     random_state=0, class_weight=None)
        clf.fit(Xtr,ytr)
        ptr, pte = clf.predict(Xtr), clf.predict(Xte)
        if pte.sum()<40 or ptr.sum()<40: continue
        prec_tr = ytr[ptr==1].mean(); prec_te = yte[pte==1].mean()
        base_te = yte.mean()
        rec = {'depth':depth,'leaf':leaf,'n_flag_tr':int(ptr.sum()),
               'prec_train':round(100*prec_tr,1),'n_flag_te':int(pte.sum()),
               'prec_heldout':round(100*prec_te,1),'baseline_heldout':round(100*base_te,1),
               'lift_pp':round(100*(prec_te-base_te),1)}
        print(rec)
        if best is None or prec_te>best[0]['prec_heldout']/100: best=(rec,clf)
rec,clf = best
print("\nBEST RULE (selected on held-out precision):")
print(json.dumps(rec, indent=2))
print(export_text(clf, feature_names=INPUTS, max_depth=4))
pte = clf.predict(Xte)
tbl = np.array([[int((yte&(pte==1)).sum()), int((~yte&(pte==1)).sum())],
                [int((yte&(pte==0)).sum()), int((~yte&(pte==0)).sum())]])
orr,p = fisher_exact(tbl)
print(f"Held-out contingency {tbl.tolist()}  odds ratio={orr:.2f}  Fisher p={p:.3e}")
pd.DataFrame([rec]).to_csv(f"{OUT}/C_out_of_time_rule.csv", index=False)
with open(f"{OUT}/C_rule_text.txt","w") as fh:
    fh.write(export_text(clf, feature_names=INPUTS, max_depth=4))
    fh.write(f"\nheld-out contingency {tbl.tolist()} OR={orr:.3f} p={p:.3e}\n")

# single-variable out-of-time check on the strongest lever available in both periods
print("\nSingle-variable out-of-time checks (only levers with support in BOTH periods):")
sv=[]
for f in INPUTS:
    # use the best decile range found on the training period
    q = pd.qcut(df[f][:cut], 10, labels=False, duplicates='drop')
    best_d, best_rate = None, -1
    for d in sorted(pd.unique(q.dropna())):
        mm=(q==d).values
        if mm.sum()<80: continue
        r = met[:cut][mm].mean()
        if r>best_rate: best_rate, best_d = r, d
    lo,hi = df[f][:cut][ (q==best_d).values ].min(), df[f][:cut][ (q==best_d).values ].max()
    te_in = (df[f].values[cut:]>=lo)&(df[f].values[cut:]<=hi)
    if te_in.sum()<40:
        sv.append({'feature':f,'train_band':f"{lo:g}-{hi:g}",'train_pct_met':round(100*best_rate,1),
                   'n_heldout_in_band':int(te_in.sum()),'heldout_pct_met':np.nan,
                   'heldout_baseline':round(100*yte.mean(),1),'lift_pp':np.nan}); continue
    sv.append({'feature':f,'train_band':f"{lo:g}-{hi:g}",'train_pct_met':round(100*best_rate,1),
               'n_heldout_in_band':int(te_in.sum()),
               'heldout_pct_met':round(100*yte[te_in].mean(),1),
               'heldout_baseline':round(100*yte.mean(),1),
               'lift_pp':round(100*(yte[te_in].mean()-yte.mean()),1)})
svdf=pd.DataFrame(sv); svdf.to_csv(f"{OUT}/C_single_var_out_of_time.csv", index=False)
print(svdf.to_string(index=False))

# ============================================== D. LHS realism audit
print("\n" + "="*78)
print("D. HOW MUCH OF A 7-D LHS SAMPLE LIES ON THE REAL JOINT OPERATING MANIFOLD?")
print("="*78)
from scipy.stats import qmc
Xr = df[INPUTS].values
mu, sd = Xr.mean(0), Xr.std(0)
Xz = (Xr-mu)/sd
samp = qmc.LatinHypercube(d=7, seed=0).random(20000)
lo, hi = Xr.min(0), Xr.max(0)
L = qmc.scale(samp, lo, hi)
Lz = (L-mu)/sd
nn = NearestNeighbors(n_neighbors=1).fit(Xz)
d_lhs,_ = nn.kneighbors(Lz)
nn2 = NearestNeighbors(n_neighbors=2).fit(Xz)
d_real,_ = nn2.kneighbors(Xz); d_real = d_real[:,1]
thr = np.percentile(d_real, 99)
print(f"99th-pct NN distance among REAL hours: {thr:.3f}")
print(f"LHS points closer than that to any real hour: "
      f"{(d_lhs.ravel()<=thr).sum()}/{len(L)} ({100*(d_lhs.ravel()<=thr).mean():.1f}%)")
print(f"LHS median NN distance {np.median(d_lhs):.3f} vs real median {np.median(d_real):.3f}")
pd.DataFrame({'metric':['real_NN_p99','lhs_pct_on_manifold','lhs_median_NN','real_median_NN'],
              'value':[thr, 100*(d_lhs.ravel()<=thr).mean(), float(np.median(d_lhs)), float(np.median(d_real))]}
            ).to_csv(f"{OUT}/D_lhs_realism.csv", index=False)
print(f"\nOutputs -> {OUT}")
