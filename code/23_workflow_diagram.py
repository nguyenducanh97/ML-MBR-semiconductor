#!/usr/bin/env python3
"""Workflow diagram of the framework.
"""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FIG = str(ROOT / "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'savefig.dpi': 600,
                     'svg.fonttype': 'none'})
BLUE, GREEN, ORANGE, RED, PURPLE = '#2E5FA3', '#1E8449', '#E67E22', '#C0392B', '#6C3FA0'
INK = '#16232E'
FILL = {BLUE: '#EBF2FA', GREEN: '#EAF6EE', ORANGE: '#FDF3E7', RED: '#FBEDED', PURPLE: '#F2EDF9'}

fig = plt.figure(figsize=(12.6, 8.2))
ax = fig.add_axes([0, 0, 1, 1])   # axes fills the canvas so 100 units = 12.6 in
ax.set_xlim(0, 100); ax.set_ylim(-1.5, 100.5); ax.axis('off')

TITLE_FS, BODY_FS = 13.5, 12.0
LINE_H = 5.0
TITLE_GAP = 6.0

def box(cx, cy, w, title, lines, ec, title_fs=TITLE_FS, body_fs=BODY_FS):
    h = TITLE_GAP + LINE_H * len(lines) + 4.0
    x, y = cx - w / 2, cy - h / 2
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0,rounding_size=1.5",
                 fc=FILL[ec], ec=ec, lw=2.0, zorder=2))
    ty = y + h - 3.8
    ax.text(cx, ty, title, ha='center', va='center', fontsize=title_fs,
            fontweight='bold', color=ec, zorder=3)
    by = ty - TITLE_GAP
    for ln in lines:
        ax.text(cx, by, ln, ha='center', va='center', fontsize=body_fs,
                color=INK, zorder=3)
        by -= LINE_H
    return x, y, w, h

def arrow(p1, p2, c='#4A5560', lw=2.2):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle='-|>', mutation_scale=17,
                 lw=lw, color=c, shrinkA=0, shrinkB=0, zorder=1))

ax.text(50, 96.5, 'Interpretable ML framework for full-scale semiconductor MBR operation',
        ha='center', va='center', fontsize=17.5, fontweight='bold', color=INK)

# ---------------- row 1
R1Y, W1 = 81, 21.6
CX1 = [13.0, 37.7, 62.3, 87.0]
box(CX1[0], R1Y, W1, '1  DATA', ['Full-scale A–O–A–O MBR', '4,593 hourly records',
                                 '7 inputs, 3 targets'], BLUE)
box(CX1[1], R1Y, W1, '2  PREPROCESSING', ['CIP periods removed', 'z-score, training fold',
                                          '7,344 h → 4,593 h'], BLUE)
box(CX1[2], R1Y, W1, '3  BENCHMARKING', ['16 models, 6 families', '48 model–target fits',
                                         'Extra Trees selected'], GREEN)
box(CX1[3], R1Y, W1, '4  VALIDATION', ['Random and 20-split', 'Blocked 24 h and 168 h',
                                       'Chronological · B-stream'], GREEN)
for a, b in zip(CX1[:-1], CX1[1:]):
    arrow((a + W1 / 2, R1Y), (b - W1 / 2, R1Y))

# ---------------- row 2
R2Y = 54
box(CX1[0], R2Y, W1, '5  INTERPRETABILITY', ['SHAP over 16 models', 'TreeSHAP 9 · Kernel 7',
                                             'Consensus + divergence'], ORANGE)
box(CX1[1], R2Y, W1, '6  SAMPLING', ['Local-covariance kernel', '500,000 states on the',
                                     'real operating manifold'], ORANGE)
box(CX1[2], R2Y, W1, '7  FEASIBILITY', ['TMP, Flow and Level', 'satisfied jointly',
                                        '207,238 feasible states'], RED)
box(CX1[3], R2Y, W1, '8  OPERATING WINDOW', ['p(targets met | x)', 'Per-variable zones',
                                             'plus a joint rule'], RED)
BH1 = TITLE_GAP + LINE_H * 3 + 4.0
ax.plot([CX1[3], CX1[3], CX1[0], CX1[0]],
        [R1Y - BH1 / 2, 68.2, 68.2, R2Y + BH1 / 2 + 1.4],
        color='#4A5560', lw=2.2, solid_capstyle='round', zorder=1)
arrow((CX1[0], R2Y + BH1 / 2 + 1.8), (CX1[0], R2Y + BH1 / 2))
for a, b in zip(CX1[:-1], CX1[1:]):
    arrow((a + W1 / 2, R2Y), (b - W1 / 2, R2Y))

# ---------------- row 3 validations
R3Y = 27
CX3 = [17.0, 50, 83.0]
box(CX3[0], R3Y, 29.5, 'A · ON THE MANIFOLD',
    ['Model rate vs measured rate', '70 operating bins,  r = 0.99'], PURPLE, 13.0, 11.5)
box(CX3[1], R3Y, 29.5, 'B · OUT OF TIME',
    ['Months 1–7 window, tested on 8–11', '39.3% → 61.9%,  p < 10⁻²⁸'], PURPLE, 13.0, 11.5)
box(CX3[2], R3Y, 29.5, 'C · PARALLEL STREAM',
    ['Independent B-stream, no refit', 'R² 0.996 / 0.980 / 0.972'], PURPLE, 13.0, 11.5)
BH2 = TITLE_GAP + LINE_H * 2 + 4.0
arrow((CX1[0], R2Y - BH1 / 2), (CX3[0], R3Y + BH2 / 2))
arrow((CX1[2], R2Y - BH1 / 2), (CX3[1], R3Y + BH2 / 2))
arrow((CX1[3], R2Y - BH1 / 2), (CX3[2], R3Y + BH2 / 2))

# ---------------- outcome
OY, h = 7.6, 12.4
ax.add_patch(FancyBboxPatch((4, OY - h / 2), 92, h,
             boxstyle="round,pad=0,rounding_size=1.5",
             fc='#E7F5EC', ec=GREEN, lw=2.2, zorder=2))
ax.text(50, OY + 3.7, 'EVIDENCE-GRADED OPERATING PROTOCOL',
        ha='center', va='center', fontsize=15.5, fontweight='bold', color=GREEN, zorder=3)
ax.text(50, OY - 0.5,
        'SRT 43–72 d · HRT 5.9–6.4 h · MLSS 5,580–6,138 mg/L · '
        'F/M 0.023–0.025 day$^{-1}$ · Air 4,868–5,892 m$^3$/h',
        ha='center', va='center', fontsize=13.5, fontweight='bold', color=INK, zorder=3)
ax.text(50, OY - 4.2, 'Target attainment inside the window 55–82% against a 37% plant-wide baseline',
        ha='center', va='center', fontsize=12.5, color='#33475B', zorder=3)
for c in CX3:
    arrow((c, R3Y - BH2 / 2), (c, OY + h / 2))

for ext in ('png', 'svg'):
    plt.savefig(f"{FIG}/Figure_1_workflow.{ext}", bbox_inches='tight',
                pad_inches=0.08, facecolor='white')
plt.close()
print("Figure 1 rebuilt (png + svg)")
