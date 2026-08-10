#!/usr/bin/env python3
"""Round each recommended band to operational precision and recompute every
statistic from the rounded bounds, so that the published numbers are
reproducible from the published band.
"""
import numpy as np, pandas as pd, json

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RES = str(ROOT / "results")
DATA = str(ROOT / "data" / "UF_data.csv")
INPUTS = ['glu', 'mlss', 'air', 'fm', 'cn', 'hrt', 'srt']
UNITS = {'glu': 'L/min', 'mlss': 'mg/L', 'air': 'm3/h', 'fm': '1/day', 'cn': '-',
         'hrt': 'h', 'srt': 'day'}
# operational precision for each variable
STEP = {'glu': 0.001, 'mlss': 1, 'air': 1, 'fm': 0.0001, 'cn': 0.01, 'hrt': 0.001, 'srt': 0.1}
DEC = {'glu': 3, 'mlss': 0, 'air': 0, 'fm': 4, 'cn': 2, 'hrt': 3, 'srt': 1}
TARGETS = {'tmp': (-0.09, -0.03), 'flow': (1.5, 2.2), 'lv': (65.0, 67.0)}

d = pd.read_csv(DATA); d.columns = [c.strip().lstrip('﻿') for c in d.columns]
n = len(d); cut = int(0.7 * n)
met = np.ones(n, bool)
for k, (lo, hi) in TARGETS.items():
    met &= (d[k].values >= lo) & (d[k].values <= hi)
T = pd.read_csv(f"{RES}/F_revised_table4_rate_based.csv")
O = pd.read_csv(f"{RES}/F_out_of_time_zones.csv")
FULL = pd.read_csv(f"{RES}/E_manifold_sample_FULL.csv")

def rnd(f, v, up):
    s = STEP[f]
    return (np.ceil(v / s) if up else np.floor(v / s)) * s

def fmt(f, v):
    return f"{v:,.{DEC[f]}f}" if f in ('mlss', 'air') else f"{v:.{DEC[f]}f}"

rows = []
print("ROUNDED BANDS, ALL STATISTICS RECOMPUTED FROM THE ROUNDED BOUNDS\n")
for f in INPUTS:
    r = T[T.Parameter == f].iloc[0]
    lo0, hi0 = [float(x) for x in r['Recommended zone'].split(' - ')]
    # round outward: the published band must contain the band that was selected,
    # so no supporting observation is silently excluded
    lo, hi = rnd(f, lo0, False), rnd(f, hi0, True)
    inw = (d[f].values >= lo) & (d[f].values <= hi)
    ms = (FULL[f].values >= lo) & (FULL[f].values <= hi)
    rows.append({'Parameter': f, 'Unit': UNITS[f],
                 'Observed range': f"{fmt(f, rnd(f, d[f].min(), False))} - {fmt(f, rnd(f, d[f].max(), True))}",
                 'Recommended zone': f"{fmt(f, lo)} - {fmt(f, hi)}",
                 'Real hours in zone': int(inw.sum()),
                 'Attainment in zone %': round(100 * met[inw].mean(), 1),
                 'Attainment outside %': round(100 * met[~inw].mean(), 1),
                 'Model rate in zone %': round(100 * FULL.feasible.values[ms].mean(), 1)})
    print(f"  {f.upper():5s} {r['Recommended zone']:>20s} -> {fmt(f, lo)} - {fmt(f, hi):<10s} "
          f"n {r['Real hours in zone']:4d} -> {inw.sum():4d}   "
          f"att {r['Attainment in zone %']:5.1f} -> {100*met[inw].mean():5.1f}%")
pd.DataFrame(rows).to_csv(f"{RES}/F_revised_table4_rate_based.csv", index=False)

# out-of-time bands rounded the same way
tr = np.arange(n) < cut; te = ~tr
oot = []
print("\nOUT-OF-TIME BANDS (rounded, recomputed)")
for f in INPUTS:
    r = O[O.parameter == f].iloc[0]
    lo0, hi0 = [float(x) for x in r['zone_from_first70'].split(' - ')]
    lo, hi = rnd(f, lo0, False), rnd(f, hi0, True)
    inw = (d[f].values >= lo) & (d[f].values <= hi)
    a_, b_ = inw & te, (~inw) & te
    lift = round(100 * (met[a_].mean() - met[te].mean()), 1) if a_.sum() >= 30 else np.nan
    oot.append({'parameter': f, 'zone_from_first70': f"{fmt(f, lo)} - {fmt(f, hi)}",
                'n_heldout_in': int(a_.sum()),
                'heldout_attain_in%': round(100 * met[a_].mean(), 1) if a_.sum() >= 30 else np.nan,
                'heldout_attain_out%': round(100 * met[b_].mean(), 1) if b_.sum() >= 30 else np.nan,
                'heldout_baseline%': round(100 * met[te].mean(), 1),
                'lift_pp': lift})
    print(f"  {f.upper():5s} {fmt(f, lo)} - {fmt(f, hi):<10s} n={int(a_.sum()):4d}  lift {lift}")
pd.DataFrame(oot).to_csv(f"{RES}/F_out_of_time_zones.csv", index=False)

# decile-profile values quoted in the discussion, recomputed on the same bins
prof = pd.read_csv(f"{RES}/F_conditional_rate_profiles.csv")
print("\nDECILE VALUES QUOTED IN THE TEXT")
for f, q in [('srt', [(76, 85), (84, 87), (86, 93)]), ('mlss', [(0, 3600), (6100, 1e9)]),
             ('air', [(6300, 1e9)]), ('fm', [(0.037, 1)]), ('cn', [(11, 13)])]:
    s = prof[prof.feature == f].dropna(subset=['real_rate'])
    for a, b in q:
        sel = s[(s.lo >= a) & (s.hi <= b)]
        if len(sel):
            print(f"  {f.upper():5s} {sel.lo.iloc[0]:.4g}-{sel.hi.iloc[0]:.4g} : {sel.real_rate.iloc[0]:.1f}%")
json.dump({'rounding': {k: STEP[k] for k in STEP},
           'note': 'bands rounded outward to the printed precision; all statistics '
                   'recomputed from the rounded bounds so published numbers are reproducible'},
          open(f"{RES}/F_rounding.json", "w"), indent=2)
print("\nwritten")
