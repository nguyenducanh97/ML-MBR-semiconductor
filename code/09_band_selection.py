#!/usr/bin/env python3
"""Operability-aware band selection. For each input the recommended band is the
widest contiguous run of deciles whose measured attainment stays within a
tolerance of the maximum, subject to a minimum support in logged hours. Bands
are then re-derived on the first 70% of the record and tested on the final 30%.
"""
import numpy as np, pandas as pd, json

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RES = str(ROOT / "results")
DATA = str(ROOT / "data" / "UF_data.csv")
INPUTS = ['glu', 'mlss', 'air', 'fm', 'cn', 'hrt', 'srt']
UNITS = {'glu': 'L/min', 'mlss': 'mg/L', 'air': 'm3/h', 'fm': '1/day', 'cn': '-',
         'hrt': 'h', 'srt': 'day'}
TARGETS = {'tmp': (-0.09, -0.03), 'flow': (1.5, 2.2), 'lv': (65.0, 67.0)}
TOL, MIN_SUPPORT = 5.0, 250

d = pd.read_csv(DATA); d.columns = [c.strip().lstrip('﻿') for c in d.columns]
n = len(d); cut = int(0.7 * n)
met = np.ones(n, bool)
for k, (lo, hi) in TARGETS.items():
    met &= (d[k].values >= lo) & (d[k].values <= hi)
P = pd.read_csv(f"{RES}/F_conditional_rate_profiles.csv")
OLD = pd.read_csv(f"{RES}/F_revised_table4_rate_based.csv")
FULL = pd.read_csv(f"{RES}/E_manifold_sample_FULL.csv")

rows, oot = [], []
for f in INPUTS:
    s = P[P.feature == f].dropna(subset=['real_rate']).sort_values('lo').reset_index(drop=True)
    best = s.real_rate.max()
    ok = (s.real_rate >= best - TOL).values
    runs, i = [], 0
    while i < len(ok):
        if ok[i]:
            j = i
            while j + 1 < len(ok) and ok[j + 1]:
                j += 1
            runs.append((i, j)); i = j + 1
        else:
            i += 1
    # widest run that also clears the support floor
    cand = []
    for a, b in runs:
        lo, hi = s.lo[a], s.hi[b]
        sup = ((d[f].values >= lo) & (d[f].values <= hi)).sum()
        if sup >= MIN_SUPPORT:
            cand.append((hi - lo, lo, hi, sup))
    if not cand:
        a, b = max(runs, key=lambda r: s.hi[r[1]] - s.lo[r[0]])
        lo, hi = s.lo[a], s.hi[b]
        sup = ((d[f].values >= lo) & (d[f].values <= hi)).sum()
    else:
        _, lo, hi, sup = max(cand)
    inw = (d[f].values >= lo) & (d[f].values <= hi)
    ms = (FULL[f].values >= lo) & (FULL[f].values <= hi)
    rows.append({'Parameter': f, 'Unit': UNITS[f],
                 'Observed range': f"{d[f].min():.4g} - {d[f].max():.4g}",
                 'Recommended zone': f"{lo:.4g} - {hi:.4g}",
                 'Real hours in zone': int(inw.sum()),
                 'Attainment in zone %': round(100 * met[inw].mean(), 1),
                 'Attainment outside %': round(100 * met[~inw].mean(), 1),
                 'Model rate in zone %': round(100 * FULL.feasible.values[ms].mean(), 1)})

# ---- proper out-of-time test: bands re-derived on the FIRST 70% only,
#      then evaluated on the final 30% which took no part in deriving them
tr = np.arange(n) < cut
te = ~tr
for f in INPUTS:
    q = pd.qcut(d[f][tr], 10, labels=False, duplicates='drop')
    prof = []
    for b in sorted(pd.unique(q.dropna())):
        m = (q == b).values
        if m.sum() < 30:
            continue
        prof.append({'lo': d[f][tr][m].min(), 'hi': d[f][tr][m].max(),
                     'rate': 100 * met[tr][m].mean()})
    pr = pd.DataFrame(prof).sort_values('lo').reset_index(drop=True)
    bestr = pr.rate.max()
    ok = (pr.rate >= bestr - TOL).values
    runs, i = [], 0
    while i < len(ok):
        if ok[i]:
            j = i
            while j + 1 < len(ok) and ok[j + 1]:
                j += 1
            runs.append((i, j)); i = j + 1
        else:
            i += 1
    cand = []
    for a, b in runs:
        lo, hi = pr.lo[a], pr.hi[b]
        sup = ((d[f].values >= lo) & (d[f].values <= hi) & tr).sum()
        if sup >= MIN_SUPPORT:
            cand.append((hi - lo, lo, hi))
    if not cand:
        a, b = max(runs, key=lambda r: pr.hi[r[1]] - pr.lo[r[0]])
        lo, hi = pr.lo[a], pr.hi[b]
    else:
        _, lo, hi = max(cand)
    inw = (d[f].values >= lo) & (d[f].values <= hi)
    a_, b_ = inw & te, (~inw) & te
    oot.append({'parameter': f, 'zone_from_first70': f"{lo:.4g} - {hi:.4g}",
                'n_heldout_in': int(a_.sum()),
                'heldout_attain_in%': round(100 * met[a_].mean(), 1) if a_.sum() >= 30 else np.nan,
                'heldout_attain_out%': round(100 * met[b_].mean(), 1) if b_.sum() >= 30 else np.nan,
                'heldout_baseline%': round(100 * met[te].mean(), 1),
                'lift_pp': round(100 * (met[a_].mean() - met[te].mean()), 1) if a_.sum() >= 30 else np.nan})

T = pd.DataFrame(rows)
T.to_csv(f"{RES}/F_revised_table4_rate_based.csv", index=False)
O = pd.DataFrame(oot)
O.to_csv(f"{RES}/F_out_of_time_zones.csv", index=False)

print("REVISED OPERATING WINDOW (operability-aware rule)\n")
comp = T.merge(OLD[['Parameter', 'Recommended zone', 'Attainment in zone %']],
               on='Parameter', suffixes=('', '_old'))
for _, r in comp.iterrows():
    ch = "unchanged" if r['Recommended zone'] == r['Recommended zone_old'] else "WIDENED"
    print(f"{r.Parameter.upper():5s} {r['Recommended zone']:>22s} {r.Unit:6s} "
          f"att {r['Attainment in zone %']:5.1f}%  (was {r['Recommended zone_old']:>18s} "
          f"at {r['Attainment in zone %_old']:.1f}%)  n={r['Real hours in zone']:4d}  {ch}")
print("\nOUT-OF-TIME TRANSFER OF THE REVISED BANDS")
print(O.to_string(index=False))
print(f"\nAttainment span across the seven bands: "
      f"{T['Attainment in zone %'].min():.0f}-{T['Attainment in zone %'].max():.0f}%")
json.dump({'rule': f'widest contiguous decile band within {TOL} pp of the maximum '
                   f'attainment, minimum support {MIN_SUPPORT} logged hours',
           'tol_pp': TOL, 'min_support_h': MIN_SUPPORT},
          open(f"{RES}/F_zone_rule.json", "w"), indent=2)
