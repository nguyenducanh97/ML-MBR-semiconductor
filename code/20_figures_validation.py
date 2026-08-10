#!/usr/bin/env python3
"""Validation figures: temporal structure and refit schedule, model-derived rate
against measured attainment, and the recommended window with its out-of-time
test.
"""
import os, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RES=str(ROOT / "results")
FIG=str(ROOT / "figures")
os.makedirs(FIG,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'axes.linewidth':1.3,
                     'axes.edgecolor':'#333333','savefig.dpi':600,'figure.dpi':110,
                     'axes.labelsize':15,'axes.titlesize':15.5,'legend.frameon':False,
                     'xtick.labelsize':13,'ytick.labelsize':13,'legend.fontsize':13.5,
                     'svg.fonttype':'none'})
BLUE,RED,GREEN,ORANGE,GREY='#2E5FA3','#C0392B','#1E8449','#E67E22','#7F8C8D'
INPUTS=['glu','mlss','air','fm','cn','hrt','srt']
LBL={'glu':'Glu (L/min)','mlss':'MLSS (mg/L)','air':'Air (m$^3$/h)',
     'fm':'F/M (day$^{-1}$)','cn':'C/N (–)','hrt':'HRT (h)','srt':'SRT (day)'}
SYM={'glu':'Glu','mlss':'MLSS','air':'Air','fm':'F/M','cn':'C/N','hrt':'HRT','srt':'SRT'}

# ============================================================ FIGURE R1  (main-text Figure 6)
sw=pd.read_csv(f"{RES}/2_block_size_sweep_summary.csv")
acf=pd.read_csv(f"{RES}/A_autocorrelation.csv")
sm=pd.read_csv(f"{RES}/F_summary_all_splits.csv")
rf=pd.read_csv(f"{RES}/H_refit_schedule.csv")
_d=pd.read_csv(str(ROOT / "data" / "UF_data.csv"))
_d.columns=[c.strip().lstrip('﻿') for c in _d.columns]
IQR={t:np.percentile(_d[t],75)-np.percentile(_d[t],25) for t in ['tmp','flow','lv']}
sw['MAE_pct_IQR']=[100*r.MAE_mean/IQR[r.target] for r in sw.itertuples()]
TRI=[('tmp',RED,'TMP'),('flow',BLUE,'Flow'),('lv',GREEN,'Level')]

fig,axz=plt.subplots(2,2,figsize=(17.5,13.2))
ax=axz.ravel()

# ---------------- (A) autocorrelation
for v,c,l in TRI:
    r=acf[acf.variable==v].iloc[0]
    ax[0].plot([1,6,24,168],[r.acf_lag1,r.acf_lag6,r.acf_lag24,r.acf_lag168],'o-',
               color=c,lw=2.8,ms=9,label=l)
for v in INPUTS:
    r=acf[acf.variable==v].iloc[0]
    ax[0].plot([1,6,24,168],[r.acf_lag1,r.acf_lag6,r.acf_lag24,r.acf_lag168],'-',
               color=GREY,lw=1.4,alpha=.55)
ax[0].plot([],[],color=GREY,lw=1.4,alpha=.55,label='7 process inputs')
ax[0].set_xscale('log'); ax[0].set_xticks([1,6,24,168]); ax[0].set_xticklabels(['1','6','24','168'])
ax[0].set_xlabel('Lag (h)'); ax[0].set_ylabel('Autocorrelation')
ax[0].axhline(1/np.e,ls=':',color='k',lw=1); ax[0].text(1.15,1/np.e+.03,'1/e',fontsize=12)
ax[0].set_title('(A) Consecutive hours resemble one another',loc='left',fontweight='bold')
ax[0].legend(loc='lower left',ncol=2,fontsize=12.5)
ax[0].set_ylim(0,1.05); ax[0].grid(alpha=.22,ls=':')

# ---------------- (B) block sweep, R2 with the absolute error alongside
for v,c,l in TRI:
    s2=sw[sw.target==v].sort_values('block_h')
    ax[1].errorbar(s2.block_h,s2.R2_mean,yerr=s2.R2_sd,fmt='o-',color=c,lw=2.8,ms=8,capsize=4,label=l)
axb=ax[1].twinx()
for v,c,l in TRI:
    s2=sw[sw.target==v].sort_values('block_h')
    axb.plot(s2.block_h,s2.MAE_pct_IQR,'--',color=c,lw=1.7,alpha=.75,marker='^',ms=6)
axb.set_ylabel('Mean absolute error (% of target IQR)',fontsize=14)
axb.set_ylim(0,100); axb.tick_params(labelsize=12.5)
ax[1].set_xscale('log'); ax[1].set_xticks([1,6,24,72,168,336,720])
ax[1].set_xticklabels(['1','6','24','72','168','336','720'])
ax[1].axhline(0,color='k',lw=.8); ax[1].set_ylim(-1.2,1.42)
ax[1].axvspan(0.72,1.42,color=ORANGE,alpha=.15)
ax[1].annotate('as published\n(random split)',xy=(1.05,1.02),xytext=(2.4,1.28),
               fontsize=11.5,color=ORANGE,ha='left',va='center',
               arrowprops=dict(arrowstyle='-',color=ORANGE,lw=1.1))
ax[1].axvline(24,color='k',ls=':',lw=1.2)
ax[1].text(27,-1.12,'operational horizon',fontsize=11.5,color='#444444')
ax[1].set_xlabel('Contiguous block length used for splitting (h)')
ax[1].set_ylabel('Test $R^2$ (Extra Trees)')
ax[1].set_title('(B) Solid, $R^2$ falls;  dashed, absolute error stays modest',
                loc='left',fontweight='bold')
_h=[plt.Line2D([],[],color=c,lw=2.8,marker='o',ms=8,label=l) for _,c,l in TRI]
_h+=[plt.Line2D([],[],color='#555555',lw=2.4,label='$R^2$, left axis'),
     plt.Line2D([],[],color='#555555',lw=1.7,ls='--',marker='^',ms=6,label='error, right axis')]
ax[1].legend(handles=_h,loc='upper right',bbox_to_anchor=(0.995,0.99),ncol=2,fontsize=11.8,
             frameon=True,framealpha=.92,edgecolor='#CCCCCC')

# ---------------- (C) four designs
sub=sw.set_index(['target','block_h'])
et=sm[sm.model=='Extra Trees'].set_index('target')
nm=['Random\n(1 h)','Blocked\n24 h','Blocked\n168 h','Chronological\n(frozen)']
x=np.arange(4); w=.26; FLOOR=-0.55
for i,(v,c,l) in enumerate(TRI):
    raw=[sub.loc[(v,1),'R2_mean'],sub.loc[(v,24),'R2_mean'],
         sub.loc[(v,168),'R2_mean'],et.loc[v,'chronological']]
    vals=[max(r,FLOOR) for r in raw]
    ax[2].bar(x+(i-1)*w,vals,w,color=c,alpha=.88,label=l,edgecolor='white')
    for xi,vv,rr in zip(x+(i-1)*w,vals,raw):
        ax[2].text(xi,vv+.035 if vv>=0 else vv-.035,f'{rr:.2f}',ha='center',
                   va='bottom' if vv>=0 else 'top',fontsize=11,rotation=90)
ax[2].set_xticks(x); ax[2].set_xticklabels(nm,fontsize=12.5); ax[2].axhline(0,color='k',lw=.8)
ax[2].set_ylabel('Test $R^2$'); ax[2].set_ylim(FLOOR-.24,1.22)
ax[2].axhline(FLOOR,color=GREY,ls=':',lw=1.1)
ax[2].text(-0.42,FLOOR-.15,'bars clipped at this line',fontsize=11,color=GREY,ha='left',style='italic')
ax[2].set_title('(C) The same model under four validation designs',loc='left',fontweight='bold')
ax[2].legend(fontsize=12.5,ncol=3,loc='upper right',frameon=True,framealpha=.92,edgecolor='#CCCCCC')
ax[2].grid(axis='y',alpha=.22,ls=':')

# ---------------- (D) deployment mode: how often the model is refitted
ORD=['frozen','monthly','fortnightly','weekly','daily']
LBLD=['Frozen\n(never)','Monthly','Fortnightly','Weekly','Daily']
xd=np.arange(5)
for v,c,l in TRI:
    s3=rf[rf.target==v].set_index('mode').reindex(ORD)
    ax[3].plot(xd,s3.R2.values,'o-',color=c,lw=3.0,ms=10,label=l)
ax[3].axhline(0,color='k',lw=.9)
ax[3].axvspan(-0.4,0.4,color=GREY,alpha=.16)
ax[3].set_xticks(xd); ax[3].set_xticklabels(LBLD,fontsize=12.5)
ax[3].set_xlim(-0.5,4.5); ax[3].set_ylim(-4.35,1.35)
ax[3].set_xlabel('How often the model is refitted on the data logged so far')
ax[3].set_ylabel('Test $R^2$ on the held-out final 30%')
ax[3].set_title('(D) Out-of-time accuracy is a question of deployment, not of the model',
                loc='left',fontweight='bold')
ax[3].legend(fontsize=12.5,ncol=1,loc='lower right',frameon=True,framealpha=.92,edgecolor='#CCCCCC')
ax[3].grid(alpha=.22,ls=':')
_t=rf[rf.target=='tmp'].set_index('mode').reindex(ORD)
ax[3].annotate(f"TMP  $R^2$ = {_t.R2.iloc[-1]:.2f},  error {_t.MAE.iloc[-1]:.4f} bar",
               xy=(3.95,_t.R2.iloc[-1]),xytext=(1.55,1.06),fontsize=12.5,color=RED,
               fontweight='bold',ha='left',va='center',
               arrowprops=dict(arrowstyle='-|>',color=RED,lw=1.6,
                               connectionstyle='arc3,rad=-0.18'))
ax[3].annotate(f"frozen for four months,\nTMP error {_t.MAE.iloc[0]:.4f} bar",
               xy=(0.04,_t.R2.iloc[0]-0.05),xytext=(0.62,-3.05),fontsize=12,color='#555555',
               ha='left',va='center',arrowprops=dict(arrowstyle='-|>',color='#888888',lw=1.4))

plt.tight_layout(h_pad=3.4,w_pad=3.6)
for e in ('png','svg'): plt.savefig(f"{FIG}/Figure_R1_temporal_validation.{e}",bbox_inches='tight',facecolor='white')
plt.close()
print("Figure R1 (2x2 with refit panel) done")

# ============================================================ FIGURE R2
P=pd.read_csv(f"{RES}/F_conditional_rate_profiles.csv").dropna(subset=['model_rate','real_rate'])
ag=json.load(open(f"{RES}/G_rate_agreement.json"))
fig=plt.figure(figsize=(20,10.5))
gs=fig.add_gridspec(2,4,hspace=.42,wspace=.32)
for i,f in enumerate(INPUTS):
    a=fig.add_subplot(gs[i//4,i%4]); s=P[P.feature==f]
    mid=(s.lo+s.hi)/2
    a.plot(mid,s.model_rate,'o-',color=BLUE,lw=2.8,ms=7,label='Model (manifold sample)')
    a.plot(mid,s.real_rate,'s--',color=RED,lw=2.8,ms=7,label='Plant record (measured)')
    a.set_xlabel(LBL[f]); a.set_ylim(-4,100)
    if i%4==0: a.set_ylabel('All-target attainment (%)')
    a.grid(alpha=.25,ls=':')
    if i==0:
        hl,ll=a.get_legend_handles_labels()
a=fig.add_subplot(gs[1,3])
a.scatter(P.real_rate,P.model_rate,s=62,c=BLUE,alpha=.75,edgecolor='white',linewidth=.8)
lim=[-3,100]; a.plot(lim,lim,'k--',lw=1.8)
a.set_xlim(lim); a.set_ylim(lim)
a.set_xlabel('Measured attainment (%)'); a.set_ylabel('Model feasibility rate (%)')
a.set_title(f"r = {ag['pearson_r']:.3f}   n = {ag['n_bins']} bins",fontsize=15,fontweight='bold')
a.grid(alpha=.25,ls=':')
fig.legend(hl,ll,loc='lower center',bbox_to_anchor=(0.5,-0.02),ncol=2,fontsize=15)
fig.suptitle('Model-derived feasibility rate reproduces the plant\'s measured target attainment',
             fontsize=17,y=.978)
for e in ('png','svg'): plt.savefig(f"{FIG}/Figure_R2_model_vs_reality.{e}",bbox_inches='tight',facecolor='white')
plt.close()
print("Figure R2 done")

# ============================================================ FIGURE R3 (main text, 2 panels)
T=pd.read_csv(f"{RES}/F_revised_table4_rate_based.csv")
O=pd.read_csv(f"{RES}/F_out_of_time_zones.csv")
jr=json.load(open(f"{RES}/F_joint_rule.json"))
fig,ax=plt.subplots(1,2,figsize=(17,6.6),gridspec_kw={'width_ratios':[1.5,1]})
y=np.arange(len(T))[::-1]
ax[0].barh(y+.2,T['Attainment in zone %'],.38,color=GREEN,label='Inside recommended zone')
ax[0].barh(y-.2,T['Attainment outside %'],.38,color=GREY,alpha=.75,label='Outside')
for yy,vi,vo in zip(y,T['Attainment in zone %'],T['Attainment outside %']):
    ax[0].text(vi+1.4,yy+.2,f'{vi:.0f}%',va='center',fontsize=13,color=GREEN,fontweight='bold')
    ax[0].text(vo+1.4,yy-.2,f'{vo:.0f}%',va='center',fontsize=12,color=GREY)
ax[0].set_yticks(y)
ax[0].set_yticklabels([f"{SYM[r.Parameter]}\n{r._4}" for r in T.itertuples()],fontsize=12)
ax[0].set_xlabel('Hours meeting all three targets (%)'); ax[0].set_xlim(0,100)
ax[0].legend(fontsize=13,loc='upper center',bbox_to_anchor=(0.5,-0.13),ncol=2)
ax[0].set_title('(A) Recommended operating window, measured effect')
ax[0].grid(axis='x',alpha=.25,ls=':')
b=ax[1].bar([0,1],[jr['heldout_baseline'],jr['heldout_precision']],.55,
            color=[GREY,GREEN],alpha=.9,edgecolor='white')
for xi,v in zip([0,1],[jr['heldout_baseline'],jr['heldout_precision']]):
    ax[1].text(xi,v+1.8,f'{v:.1f}%',ha='center',fontsize=17,fontweight='bold',
               color=GREEN if v>50 else '#5A6672')
ax[1].set_xticks([0,1])
ax[1].set_xticklabels(['All held-out\nhours','Hours inside\nthe learned window'],fontsize=13)
ax[1].set_ylabel('Hours meeting all three targets (%)'); ax[1].set_ylim(0,80)
ax[1].set_title('(B) Window learned on months 1–7,\nevaluated on months 8–11')
ax[1].text(.5,46,f"OR = {jr['odds_ratio']:.2f}\np = 3 × 10$^{{-29}}$\nn = {jr['n_flagged_heldout']} h",
           ha='center',va='center',fontsize=13,
           bbox=dict(boxstyle='round,pad=.45',fc='#F4F7FB',ec=BLUE))
ax[1].grid(axis='y',alpha=.25,ls=':')
plt.tight_layout()
for e in ('png','svg'): plt.savefig(f"{FIG}/Figure_R3_operating_window.{e}",bbox_inches='tight',facecolor='white')
plt.close()
print("Figure R3 done")

# ============================================================ SI FIGURE S17 (per-lever transfer)
Ov=O.dropna(subset=['lift_pp']).sort_values('lift_pp')
fig,axs=plt.subplots(figsize=(9.5,6.0))
cols=[GREEN if v>0 else '#D98880' for v in Ov.lift_pp]
axs.barh(np.arange(len(Ov)),Ov.lift_pp,color=cols,alpha=.9)
for i,v in enumerate(Ov.lift_pp):
    axs.text(v+(1.1 if v>0 else -1.1),i,f'{v:+.1f}',va='center',
             ha='left' if v>0 else 'right',fontsize=13,fontweight='bold')
axs.set_yticks(np.arange(len(Ov))); axs.set_yticklabels([SYM[p] for p in Ov.parameter],fontsize=13)
axs.axvline(0,color='k',lw=1.2)
axs.set_xlabel('Change in target attainment on the held-out final 30% (percentage points)')
axs.set_title('Out-of-time transfer of each individual operating band')
axs.grid(axis='x',alpha=.25,ls=':'); axs.set_xlim(-28,36)
plt.tight_layout()
for e in ('png','svg'): plt.savefig(f"{FIG}/Figure_S17_per_lever_transfer.{e}",bbox_inches='tight',facecolor='white')
plt.close()
print("Figure S17 done")
print("All figures ->",FIG)
