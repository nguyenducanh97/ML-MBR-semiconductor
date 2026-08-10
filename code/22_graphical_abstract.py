#!/usr/bin/env python3
"""Graphical abstract.
"""
import os, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RES=str(ROOT / "results")
FIG=str(ROOT / "figures")
plt.rcParams.update({'font.family':'DejaVu Sans','savefig.dpi':600,'svg.fonttype':'none',
                     'axes.linewidth':1.0,'legend.frameon':False})
BLUE,RED,GREEN,ORANGE,PURPLE,GREY='#2E5FA3','#C0392B','#1E8449','#E67E22','#6C3FA0','#7F8C8D'
INK='#1B2A38'

T=pd.read_csv(f"{RES}/F_revised_table4_rate_based.csv")
P=pd.read_csv(f"{RES}/F_conditional_rate_profiles.csv").dropna(subset=['model_rate','real_rate'])
jr=json.load(open(f"{RES}/F_joint_rule.json"))
ag=json.load(open(f"{RES}/G_rate_agreement.json"))

fig=plt.figure(figsize=(19.05/2.54*2, 9.35/2.54*2))   # ~2x Elsevier GA aspect
fig.patch.set_facecolor('white')
ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,190); ax.set_ylim(0,93); ax.axis('off')

def rbox(x,y,w,h,fc,ec,lw=1.6,r=2.0,alpha=1.0,z=1):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0,rounding_size={r}",
                 fc=fc,ec=ec,lw=lw,alpha=alpha,zorder=z))
def arw(x1,y1,x2,y2,c=INK,lw=2.0,ms=18):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=ms,
                 lw=lw,color=c,shrinkA=0,shrinkB=0,zorder=5))

# ---------------- title
ax.text(95,88.6,'Interpretable machine learning defines the operational basin of a',
        ha='center',va='center',fontsize=16.5,color=INK,fontweight='bold')
ax.text(95,84.2,'full-scale semiconductor-wastewater membrane bioreactor',
        ha='center',va='center',fontsize=16.5,color=INK,fontweight='bold')

# ================= PANEL 1 : plant + data
rbox(3,44,42,36,'#F2F7FC',BLUE,1.8)
ax.text(24,76.6,'FULL-SCALE MBR  ·  4,593 h SCADA',ha='center',fontsize=11.3,
        color=BLUE,fontweight='bold')
# schematic reactor
zx=[6.5,15.0,23.5,32.0]; zl=['ANOX','OXIC','ANOX','OXIC']
for i,(x,l) in enumerate(zip(zx,zl)):
    fc='#DCE9F7' if 'ANOX' in l else '#CFE3F5'
    ax.add_patch(Rectangle((x,58),8.0,10.5,fc=fc,ec=BLUE,lw=1.3,zorder=2))
    ax.text(x+4.0,63.2,l,ha='center',va='center',fontsize=8.2,color=INK,fontweight='bold')
ax.add_patch(Rectangle((40.0,58),4.0,10.5,fc='#E9F3EA',ec=GREEN,lw=1.4,zorder=2))
for k in range(5):
    ax.plot([40.7+k*0.65,40.7+k*0.65],[59.0,67.5],color=GREEN,lw=1.5,zorder=3)
ax.text(42.0,56.2,'UF',ha='center',fontsize=8.6,color=GREEN,fontweight='bold')
ax.annotate('',xy=(6.4,63.2),xytext=(3.9,63.2),
            arrowprops=dict(arrowstyle='-|>',color=INK,lw=1.6))
ax.text(4.6,65.0,'in',fontsize=7.6,color=INK)
ax.annotate('',xy=(44.6,63.2),xytext=(44.1,63.2),
            arrowprops=dict(arrowstyle='-|>',color=GREEN,lw=1.6))
# bubbles
rng=np.random.RandomState(3)
for _ in range(26):
    ax.add_patch(Circle((40.2+rng.rand()*3.6,58.6+rng.rand()*9.4),0.30,
                 fc='white',ec=GREEN,lw=.7,alpha=.9,zorder=4))
# inputs / outputs
ax.text(6,53.4,'7 operating inputs',fontsize=9.6,color=INK,fontweight='bold')
ax.text(6,50.4,'SRT · HRT · MLSS · F/M\nAir · C/N · Glu',fontsize=8.9,color='#33475B',
        va='top',linespacing=1.5)
ax.text(24.8,53.4,'3 membrane targets',fontsize=8.9,color=RED,fontweight='bold')
ax.text(24.8,50.4,'TMP · Flow · Level',fontsize=8.3,color='#33475B',
        va='top',linespacing=1.5)

# ================= PANEL 2 : models
rbox(49,44,40,36,'#F3FAF5',GREEN,1.8)
ax.text(69,76.6,'16 ALGORITHMS  ·  6 FAMILIES',ha='center',fontsize=11.3,
        color=GREEN,fontweight='bold')
fams=[('Ensemble tree',9),('Regularised linear',3),('Linear',1),
      ('Kernel SVR',1),('Instance KNN',1),('Neural MLP',1)]
yy=71.5
for nm,c in fams:
    ax.text(52.5,yy,nm,fontsize=8.7,color='#33475B',va='center')
    for k in range(c):
        ax.add_patch(Rectangle((70.5+k*1.85,yy-.85),1.45,1.7,
                     fc=GREEN if nm=='Ensemble tree' else '#9DC3A8',ec='none',zorder=3))
    yy-=3.0
ax.plot([52.5,86],[52.6,52.6],color=GREEN,lw=1.0,alpha=.5)
ax.text(69,50.0,'Extra Trees selected',ha='center',fontsize=10.6,color=GREEN,fontweight='bold')
ax.text(69,46.4,'validated four ways: random · blocked 24/168 h\nchronological · independent parallel UF stream',
        ha='center',fontsize=8.6,color='#33475B',linespacing=1.5)

# ================= PANEL 3 : SHAP driver
rbox(93,44,42,36,'#FEF6EC',ORANGE,1.8)
ax.text(114,76.6,'WHAT DRIVES THE MEMBRANE',ha='center',fontsize=11.3,
        color=ORANGE,fontweight='bold')
feats=['SRT','HRT','MLSS','F/M','Glu','Air','C/N']
vals=[0.53,0.143,0.144,0.085,0.070,0.063,0.041]
cols=[RED]+[ORANGE]*2+['#E8B27A']*4
yy=72.0
for f,v,c in zip(feats,vals,cols):
    ax.add_patch(Rectangle((104,yy-1.05),v*46,2.1,fc=c,ec='none',zorder=3))
    ax.text(103,yy,f,ha='right',va='center',fontsize=8.9,color=INK,
            fontweight='bold' if f=='SRT' else 'normal')
    yy-=3.35
ax.text(114,47.2,'SRT ranks first for TMP in all 16 models\nand under every validation design',
        ha='center',fontsize=9.0,color='#33475B',linespacing=1.5,fontweight='bold')
ax.text(114,44.9,'mean |SHAP|, transmembrane pressure',ha='center',fontsize=7.6,color=GREY,style='italic')

# ================= PANEL 4 : operating window
rbox(3,6,86,33,'#FBF1F1',RED,1.8)
ax.text(46,35.5,'THE OPERATIONAL BASIN  ,   where the plant actually performs',
        ha='center',fontsize=11.3,color=RED,fontweight='bold')
order=['srt','hrt','mlss','fm','air']
nice={'srt':'SRT  43–72 d','hrt':'HRT  5.9–6.4 h','mlss':'MLSS  5,580–6,138 mg/L',
      'fm':'F/M  0.023–0.025 d$^{-1}$','air':'Air  4,868–5,892 m$^3$/h'}
yy=31.0
for k in order:
    r=T[T.Parameter==k].iloc[0]
    ax.text(6,yy,nice[k],fontsize=9.2,color=INK,va='center')
    ax.add_patch(Rectangle((44,yy+0.15),r['Attainment in zone %']*0.36,1.55,
                 fc=GREEN,ec='none',zorder=4,alpha=.95))
    ax.add_patch(Rectangle((44,yy-1.55),r['Attainment outside %']*0.36,1.55,
                 fc='#C9CFD4',ec='none',zorder=3))
    ax.text(44+r['Attainment in zone %']*0.36+1.2,yy+0.92,f"{r['Attainment in zone %']:.0f}%",
            va='center',fontsize=8.8,color=GREEN,fontweight='bold',zorder=5)
    ax.text(44+r['Attainment outside %']*0.36+1.2,yy-0.78,f"{r['Attainment outside %']:.0f}%",
            va='center',fontsize=8.0,color='#7A848C',zorder=5)
    yy-=4.3
ax.add_patch(Rectangle((44,10.6),3.6,1.7,fc=GREEN,ec='none'))
ax.text(48.6,11.45,'inside the recommended zone',fontsize=8.3,va='center',color='#33475B')
ax.add_patch(Rectangle((44,7.8),3.6,1.7,fc='#C9CFD4',ec='none'))
ax.text(48.6,8.65,'outside  (plant-wide baseline 37%)',fontsize=8.3,va='center',color='#33475B')
ax.text(6,11.6,'Hours meeting TMP, flow and\nlevel targets simultaneously',fontsize=8.4,
        color=GREY,va='top',linespacing=1.5,style='italic')

# ================= PANEL 5 : validation
rbox(93,6,42,33,'#F4F0FA',PURPLE,1.8)
ax.text(114,35.5,'VALIDATED OUT OF TIME',ha='center',fontsize=11.3,color=PURPLE,fontweight='bold')
ax.text(114,32.0,'window learned on months 1–7\ntested on months 8–11',ha='center',
        fontsize=8.9,color='#33475B',linespacing=1.5)
bx=[103.5,118.5]; bv=[jr['heldout_baseline'],jr['heldout_precision']]
bc=['#C9CFD4',GREEN]; bl=['all held-out\nhours','inside the\nlearned rule']
for x,v,c,l in zip(bx,bv,bc,bl):
    ax.add_patch(Rectangle((x,14.2),9.5,v*0.185,fc=c,ec='none',zorder=3))
    ax.text(x+4.75,14.6+v*0.185,f'{v:.1f}%',ha='center',fontsize=11.4,
            color=GREEN if c==GREEN else '#5A6672',fontweight='bold')
    ax.text(x+4.75,13.3,l,ha='center',va='top',fontsize=8.2,color='#33475B',linespacing=1.4)
ax.text(114,8.2,f"odds ratio {jr['odds_ratio']:.2f}   ·   p < 10$^{{-28}}$   ·   n = {jr['n_flagged_heldout']} h",
        ha='center',fontsize=8.7,color=PURPLE,fontweight='bold')

# ================= panel 6 : agreement badge
rbox(139,44,48,36,'#F2F7FC',BLUE,1.8)
ax.text(163,76.6,'MODEL MATCHES THE PLANT',ha='center',fontsize=11.3,color=BLUE,fontweight='bold')
axi=fig.add_axes([0.775,0.567,0.165,0.205]); axi.set_facecolor('white')
axi.scatter(P.real_rate,P.model_rate,s=22,c=BLUE,alpha=.75,edgecolor='white',linewidth=.5)
axi.plot([0,100],[0,100],'k--',lw=1.0)
axi.set_xlim(-3,100); axi.set_ylim(-3,100)
axi.set_xticks([0,50,100]); axi.set_yticks([0,50,100])
axi.set_xlabel('measured attainment (%)',fontsize=7.8,labelpad=1.5)
axi.set_ylabel('model rate (%)',fontsize=7.8,labelpad=1.5)
axi.tick_params(labelsize=7.0,pad=1.5); axi.grid(alpha=.25,ls=':')
for s in axi.spines.values(): s.set_color('#8FA8C4')
ax.text(163,47.0,f"r = {ag['pearson_r']:.2f} across {ag['n_bins']} operating bins",
        ha='center',fontsize=9.8,color=BLUE,fontweight='bold')
ax.text(163,44.8,'predicted feasibility vs measured attainment',ha='center',
        fontsize=7.7,color=GREY,style='italic')

# ================= panel 7 : outcome strip
rbox(139,6,48,33,'#F3FAF5',GREEN,1.8)
ax.text(163,35.5,'WHY IT MATTERS',ha='center',fontsize=11.3,color=GREEN,fontweight='bold')
bullets=[('Reactive → proactive','cleaning triggered by alarms is replaced by\noperation inside a data-derived basin'),
         ('Three targets at once','pressure, throughput and hydraulic stability\nreconciled rather than traded off'),
         ('Transferable method','manifold-constrained feasibility mapping\napplies to any multi-output plant record')]
yy=31.0
for h,b in bullets:
    ax.add_patch(Circle((143.0,yy+.35),0.75,fc=GREEN,ec='none',zorder=3))
    ax.text(145.4,yy+.35,h,fontsize=9.3,color=INK,fontweight='bold',va='center')
    ax.text(145.4,yy-1.4,b,fontsize=8.2,color='#33475B',va='top',linespacing=1.45)
    yy-=8.6

# flow arrows
arw(45.4,62,48.6,62); arw(89.4,62,92.6,62); arw(135.4,62,138.6,62)
arw(114,43.6,114,39.4,PURPLE); arw(46,43.6,46,39.4,RED)

for e in ('png','svg'):
    plt.savefig(f"{FIG}/Figure_0_Graphical_Abstract.{e}",bbox_inches='tight',facecolor='white')
plt.close()
print("Graphical abstract written")
