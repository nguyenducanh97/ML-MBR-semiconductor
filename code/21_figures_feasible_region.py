#!/usr/bin/env python3
"""Feasible-region figures: per-input attainment against the recommended band,
the pairwise conditional-rate map, and predicted output response across the
feasible region.
"""
import os, warnings, itertools
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RES = str(ROOT / "results")
FIG = str(ROOT / "figures")
DATA = str(ROOT / "data" / "UF_data.csv")
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 14, 'savefig.dpi': 600,
                     'axes.edgecolor': '#333333', 'axes.linewidth': 1.3,
                     'axes.labelsize': 15, 'axes.titlesize': 15,
                     'xtick.labelsize': 13, 'ytick.labelsize': 13,
                     'legend.fontsize': 13.5, 'legend.frameon': False,
                     'svg.fonttype': 'none'})
BLUE, RED, GREEN, PURPLE, ORANGE = '#2E5FA3', '#C0392B', '#1E8449', '#6C3FA0', '#E67E22'
INPUTS = ['glu', 'mlss', 'air', 'fm', 'cn', 'hrt', 'srt']
LBL = {'glu': 'Glu (L/min)', 'mlss': 'MLSS (mg/L)', 'air': 'Air (m$^3$/h)',
       'fm': 'F/M (day$^{-1}$)', 'cn': 'C/N (–)', 'hrt': 'HRT (h)', 'srt': 'SRT (day)'}
SYM = {'glu': 'Glu', 'mlss': 'MLSS', 'air': 'Air', 'fm': 'F/M', 'cn': 'C/N',
       'hrt': 'HRT', 'srt': 'SRT'}
TARGETS = {'tmp': (-0.09, -0.03), 'flow': (1.5, 2.2), 'lv': (65.0, 67.0)}

FULL = pd.read_csv(f"{RES}/E_manifold_sample_FULL.csv")
F = FULL[FULL.feasible]
P = pd.read_csv(f"{RES}/F_conditional_rate_profiles.csv")
T = pd.read_csv(f"{RES}/F_revised_table4_rate_based.csv")
print(f"full sample {len(FULL):,}   feasible {len(F):,} ({100*len(F)/len(FULL):.1f}%)")

def zone(f):
    s = T[T.Parameter == f].iloc[0]['Recommended zone']
    return [float(x.replace(',', '')) for x in s.split(' - ')]

def save(name):
    for e in ('png', 'svg'):
        plt.savefig(f"{FIG}/{name}.{e}", bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"{name} done")

# ==================================================== Figure 8
# The attainment rate is a BINNED quantity over unequal-width deciles (up to 863x
# width ratio for SRT), so it is drawn as a step across each bin's true extent
# rather than as a marker at the bin midpoint, which would misrepresent which
# part of the axis each rate applies to.
fig, axes = plt.subplots(2, 4, figsize=(23, 10.2))
for i, f in enumerate(INPUTS):
    a = axes[i // 4, i % 4]
    v = F[f].values
    lo, hi = zone(f)
    a.axvspan(lo, hi, color=GREEN, alpha=.20, zorder=0)
    a.axvline(lo, color=GREEN, lw=1.4, alpha=.7, zorder=1)
    a.axvline(hi, color=GREEN, lw=1.4, alpha=.7, zorder=1)

    # feasible-point density
    kde = gaussian_kde(v[np.random.RandomState(0).choice(len(v), min(15000, len(v)), replace=False)])
    g = np.linspace(v.min(), v.max(), 300); d = kde(g)
    a.fill_between(g, d, color=BLUE, alpha=.20, zorder=2)
    a.plot(g, d, color=BLUE, lw=2.6, zorder=3)
    a.set_xlabel(LBL[f]); a.set_yticks([])
    if i % 4 == 0: a.set_ylabel('Feasible-point density', color=BLUE)
    a.set_xlim(v.min(), v.max())

    # measured attainment, as a step over the true bin edges
    srow = P[P.feature == f].dropna(subset=['real_rate']).sort_values('lo').reset_index(drop=True)
    edges = np.concatenate([srow.lo.values, [srow.hi.values[-1]]])
    a2 = a.twinx()
    a2.stairs(srow.real_rate.values, edges, color=RED, lw=2.8, zorder=5)
    a2.stairs(srow.real_rate.values, edges, color=RED, alpha=.13, fill=True, zorder=4)
    # mark the best bin
    b = srow.loc[srow.real_rate.idxmax()]
    a2.plot([b.lo, b.hi], [b.real_rate, b.real_rate], color='#7B241C', lw=4.2, zorder=6)
    a2.set_ylim(0, 100); a2.set_xlim(v.min(), v.max())
    a2.tick_params(axis='y', labelcolor=RED, labelsize=12)
    if i % 4 == 3: a2.set_ylabel('Measured attainment (%)', color=RED, fontsize=14)
    a.set_title(f"{SYM[f]}    recommended {lo:g}–{hi:g}    {b.real_rate:.0f}% attainment",
                fontsize=13.5, color=GREEN, fontweight='bold')

lg = axes[1, 3]; lg.axis('off')
from matplotlib.lines import Line2D as _L2
from matplotlib.patches import Patch as _P2
lg.legend(handles=[
    _L2([], [], color=BLUE, lw=2.8, label='Feasible-point density (KDE)'),
    _L2([], [], color=RED, lw=2.8, label='Measured attainment rate (per decile)'),
    _L2([], [], color='#7B241C', lw=4.2, label='Highest-attainment decile'),
    _P2(facecolor=GREEN, alpha=.20, label='Recommended zone'),
], loc='center', fontsize=15)
plt.tight_layout()
save("Figure_8_feasible_ranges")

# ==================================================== Figure 9
# Plot the 2-D CONDITIONAL FEASIBILITY RATE, not the density of feasible points.
# Table 4 is derived from the rate, so the rate is the quantity the recommended
# bands should align with. Density instead reflects how long the plant happened to
# operate somewhere, which is why it sits elsewhere.
from matplotlib.colors import Normalize
import itertools

pairs = list(itertools.combinations(INPUTS, 2))
NB, MINC = 18, 20
fig, axes = plt.subplots(3, 7, figsize=(29, 12.8),
                         gridspec_kw={'wspace': 0.46, 'hspace': 0.36})
im = None
for k, (a_, b_) in enumerate(pairs):
    ax = axes[k // 7, k % 7]
    xe = np.linspace(FULL[a_].min(), FULL[a_].max(), NB + 1)
    ye = np.linspace(FULL[b_].min(), FULL[b_].max(), NB + 1)
    Hall, _, _ = np.histogram2d(FULL[a_], FULL[b_], bins=[xe, ye])
    Hfea, _, _ = np.histogram2d(F[a_], F[b_], bins=[xe, ye])
    with np.errstate(invalid='ignore', divide='ignore'):
        rate = 100.0 * Hfea / Hall
    rate[Hall < MINC] = np.nan
    cmap = plt.get_cmap('RdYlGn').copy(); cmap.set_bad('#F4F4F4')
    im = ax.pcolormesh(xe, ye, np.ma.masked_invalid(rate.T), cmap=cmap,
                       norm=Normalize(0, 100), shading='auto', zorder=1)
    # 80% rate contour: the high-performance core
    xc, yc = 0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:])
    with np.errstate(invalid='ignore'):
        Z = np.nan_to_num(rate.T, nan=0.0)
    if np.nanmax(Z) > 80:
        ax.contour(xc, yc, Z, levels=[80], colors='#0B3D2E', linewidths=2.2, zorder=4)
    # recommended bands, drawn as two crossing strips with the intersection outlined
    za, zb = zone(a_), zone(b_)
    NAVY = '#12305C'
    ax.axvspan(za[0], za[1], facecolor=NAVY, alpha=.10, zorder=3)
    ax.axhspan(zb[0], zb[1], facecolor=NAVY, alpha=.10, zorder=3)
    for x in za:
        ax.axvline(x, color=NAVY, lw=1.8, ls='--', alpha=.95, zorder=5)
    for y in zb:
        ax.axhline(y, color=NAVY, lw=1.8, ls='--', alpha=.95, zorder=5)
    ax.add_patch(plt.Rectangle((za[0], zb[0]), za[1] - za[0], zb[1] - zb[0],
                               facecolor=NAVY, alpha=.10, zorder=4))
    ax.add_patch(plt.Rectangle((za[0], zb[0]), za[1] - za[0], zb[1] - zb[0],
                               fill=False, ec=NAVY, lw=3.0, zorder=6))
    ax.set_xlim(xe[0], xe[-1]); ax.set_ylim(ye[0], ye[-1])
    ax.set_xlabel(LBL[a_], fontsize=13); ax.set_ylabel(LBL[b_], fontsize=13)
    ax.tick_params(labelsize=11)
    m = ((FULL[a_] >= za[0]) & (FULL[a_] <= za[1]) &
         (FULL[b_] >= zb[0]) & (FULL[b_] <= zb[1]))
    if m.sum() > 50:
        r_in = 100 * FULL.feasible[m].mean()
        ax.set_title(f"{r_in:.0f}% inside the box", fontsize=12.5, fontweight='bold',
                     color='#0B3D2E' if r_in >= 60 else '#B03A2E')
cb = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.013, pad=0.018)
cb.set_label('Probability all three targets are met (%)', fontsize=14)
cb.ax.tick_params(labelsize=12)
from matplotlib.lines import Line2D as _L
from matplotlib.patches import Patch as _P
fig.legend(handles=[_P(facecolor='#12305C', alpha=.10, ec='#12305C', ls='--', lw=1.8,
                       label='Recommended band for each variable'),
                    _P(facecolor='#12305C', alpha=.25, ec='#12305C', lw=3.0,
                       label='Where the two bands cross'),
                    _L([], [], color='#0B3D2E', lw=2.2, label='80% feasibility contour'),
                    _P(facecolor='#F4F4F4', ec='#B0B0B0',
                       label='Fewer than 20 sampled states, rate not estimated')],
           loc='lower center', ncol=4, fontsize=14, bbox_to_anchor=(0.47, -0.055))
fig.suptitle('Probability of meeting all three performance targets across every pair of operating '
             'variables', fontsize=17, y=.997)
save("Figure_9_pairwise_density")

# ==================================================== Figure 10
panels = [('srt', 'pred_tmp', 'pred_lv'), ('hrt', 'pred_flow', 'pred_tmp'),
          ('air', 'pred_tmp', 'pred_flow'), ('mlss', 'pred_tmp', 'pred_lv'),
          ('fm', 'pred_lv', 'pred_flow'), ('cn', 'pred_tmp', 'pred_flow')]
NICE = {'pred_tmp': 'TMP (bar)', 'pred_flow': 'Flow (m$^3$/min)',
        'pred_lv': 'Level (%)'}
KEY = {'pred_tmp': 'tmp', 'pred_flow': 'flow', 'pred_lv': 'lv'}
CC = {'pred_tmp': RED, 'pred_flow': BLUE, 'pred_lv': GREEN}

fig, axes = plt.subplots(2, 3, figsize=(21, 11))
for k, (f, o1, o2) in enumerate(panels):
    ax = axes[k // 3, k % 3]
    edges = np.unique(np.percentile(FULL[f], np.linspace(0, 100, 21)))
    mid = (edges[:-1] + edges[1:]) / 2
    for o, axis, style in [(o1, ax, 'o-'), (o2, ax.twinx(), 's--')]:
        m = np.array([FULL[o][(FULL[f] >= a) & (FULL[f] < b)].mean()
                      for a, b in zip(edges[:-1], edges[1:])])
        q1 = np.array([FULL[o][(FULL[f] >= a) & (FULL[f] < b)].quantile(.25)
                       for a, b in zip(edges[:-1], edges[1:])])
        q3 = np.array([FULL[o][(FULL[f] >= a) & (FULL[f] < b)].quantile(.75)
                       for a, b in zip(edges[:-1], edges[1:])])
        axis.fill_between(mid, q1, q3, color=CC[o], alpha=.15, zorder=1)
        axis.plot(mid, m, style, color=CC[o], lw=3.0, ms=7, zorder=3)
        lo, hi = TARGETS[KEY[o]]
        axis.axhline(lo, color=CC[o], ls=':', lw=1.8, alpha=.8)
        axis.axhline(hi, color=CC[o], ls=':', lw=1.8, alpha=.8)
        axis.set_ylabel(NICE[o], color=CC[o], fontsize=15)
        axis.tick_params(axis='y', labelcolor=CC[o], labelsize=13)
    ax.set_xlabel(LBL[f], fontsize=15)
    zl, zh = zone(f)
    ax.axvspan(zl, zh, color=GREEN, alpha=.16, zorder=0)
    ax.grid(alpha=.22, ls=':')
    ax.set_title(f"({'ABCDEF'[k]})  {SYM[f]}", fontsize=16, loc='left', fontweight='bold')
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
handles = [Line2D([], [], color=RED, lw=3, marker='o', ms=8, label='Predicted TMP'),
           Line2D([], [], color=BLUE, lw=3, marker='s', ls='--', ms=8, label='Predicted Flow (permeate flow)'),
           Line2D([], [], color=GREEN, lw=3, marker='o', ms=8, label='Predicted Level (water level)'),
           Line2D([], [], color='#555555', ls=':', lw=2, label='Performance target limits'),
           Patch(facecolor=GREEN, alpha=.16, label='Recommended operating zone'),
           Patch(facecolor='#888888', alpha=.18, label='Interquartile range across the sample')]
fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=14.5,
           bbox_to_anchor=(0.5, -0.055))
fig.suptitle('Predicted output response across the full manifold-constrained sample; '
             'dotted lines are the performance target limits',
             fontsize=17, y=.997)
plt.tight_layout(rect=[0, 0, 1, .968])
save("Figure_10_effects")
