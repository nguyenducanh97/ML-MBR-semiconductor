#!/usr/bin/env python3
"""Model-free audit of an operating window: joint support in the logged record,
per-variable attainment rates and the effect of each band in isolation.
"""
import os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RANDOM_SEED=42
DATA=str(ROOT / "data" / "UF_data.csv")
OUT=str(ROOT / "results")
os.makedirs(OUT,exist_ok=True)
INPUTS=['glu','mlss','air','fm','cn','hrt','srt']; OUTPUTS=['tmp','flow','lv']
df=pd.read_csv(DATA); df.columns=[c.strip().lstrip('﻿') for c in df.columns]
n=len(df); cut=int(0.7*n)
# ================================================== PART B - operating window
print("\n" + "=" * 78)
print("PART B. MODEL-FREE VALIDATION OF THE RECOMMENDED OPERATING WINDOW")
print("=" * 78)

# Performance targets used in the manuscript's feasibility analysis
TARGETS = {'tmp': (-0.09, -0.03), 'flow': (1.5, 2.2), 'lv': (65.0, 67.0)}
met = np.ones(n, bool)
for k, (lo, hi) in TARGETS.items():
    met &= (df[k].values >= lo) & (df[k].values <= hi)
print(f"Hours meeting all three targets over the full record: {met.sum()} / {n} "
      f"({100*met.mean():.1f}%)")

# Recommended preferred-density zone from manuscript Table 4
WINDOW = {'srt': (50, 70), 'hrt': (6.0, 7.0), 'air': (5500, 6500),
          'mlss': (3500, 6000), 'cn': (6, 8), 'fm': (0.020, 0.040), 'glu': (0.5, 0.9)}

def in_window(keys):
    m = np.ones(n, bool)
    for k in keys:
        lo, hi = WINDOW[k]
        m &= (df[k].values >= lo) & (df[k].values <= hi)
    return m

periods = {'full_record': np.ones(n, bool),
           'train_period_first70pct': np.arange(n) < cut,
           'HELD_OUT_final30pct': np.arange(n) >= cut}

rows = []
combos = {
    'SRT only (50-70 d)':                 ['srt'],
    'HRT only (6.0-7.0 h)':               ['hrt'],
    'F/M only (0.020-0.040 /d)':          ['fm'],
    'SRT + HRT':                          ['srt', 'hrt'],
    'SRT + HRT + F/M':                    ['srt', 'hrt', 'fm'],
    'SRT + HRT + F/M + Air':              ['srt', 'hrt', 'fm', 'air'],
    'All 7 recommended ranges':           list(WINDOW.keys()),
}
for pname, pmask in periods.items():
    base = met[pmask].mean()
    for cname, keys in combos.items():
        w = in_window(keys) & pmask
        o = (~in_window(keys)) & pmask
        rows.append({
            'period': pname, 'window': cname,
            'n_in_window': int(w.sum()),
            'pct_met_INSIDE': round(100*met[w].mean(), 1) if w.sum() else np.nan,
            'n_outside': int(o.sum()),
            'pct_met_OUTSIDE': round(100*met[o].mean(), 1) if o.sum() else np.nan,
            'baseline_pct': round(100*base, 1),
            'lift_pp': round(100*(met[w].mean() - met[o].mean()), 1) if (w.sum() and o.sum()) else np.nan,
        })
win = pd.DataFrame(rows)
win.to_csv(f"{OUT}/B_operating_window_validation.csv", index=False)
for pname in periods:
    print(f"\n--- {pname} ---")
    print(win[win.period == pname].drop(columns='period').to_string(index=False))

# mean outcome inside vs outside the full recommended window, held-out period
print("\n--- Mean measured outcome, HELD-OUT final 30%, all-7 window ---")
hm = periods['HELD_OUT_final30pct']
w = in_window(list(WINDOW.keys())) & hm
o = (~in_window(list(WINDOW.keys()))) & hm
out_rows = []
for k in OUTPUTS:
    out_rows.append({'variable': k,
                     'inside_mean': round(float(df[k].values[w].mean()), 4) if w.sum() else np.nan,
                     'outside_mean': round(float(df[k].values[o].mean()), 4),
                     'n_inside': int(w.sum()), 'n_outside': int(o.sum())})
om = pd.DataFrame(out_rows); om.to_csv(f"{OUT}/B_window_outcomes_heldout.csv", index=False)
print(om.to_string(index=False))

# statistical test on the held-out period, most useful practical combo
from scipy.stats import fisher_exact, chi2_contingency
print("\n--- Significance, HELD-OUT period, SRT+HRT+F/M window ---")
w = in_window(['srt', 'hrt', 'fm']) & hm
o = (~in_window(['srt', 'hrt', 'fm'])) & hm
table = np.array([[met[w].sum(), (~met[w]).sum()], [met[o].sum(), (~met[o]).sum()]])
print("contingency [[in&met, in&miss],[out&met, out&miss]] =", table.tolist())
if table.min() >= 0 and table.sum() > 0 and w.sum() > 0 and o.sum() > 0:
    try:
        orr, p = fisher_exact(table)
        print(f"Fisher exact: odds ratio = {orr:.3f}, p = {p:.3e}")
    except Exception as e:
        print("fisher failed", e)
    try:
        c2, p2, _, _ = chi2_contingency(table)
        print(f"Chi2 = {c2:.2f}, p = {p2:.3e}")
    except Exception as e:
        print("chi2 failed", e)

print(f"\nOutputs -> {OUT}")
