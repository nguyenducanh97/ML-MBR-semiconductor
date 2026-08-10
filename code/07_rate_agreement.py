#!/usr/bin/env python3
"""Agreement between the model-derived feasibility rate and the measured
attainment rate bin by bin, stability of the SHAP importance ranking across
validation designs, and pairwise correlations of the record.
"""
import os, warnings, json, time
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.base import clone
from scipy.stats import pearsonr, spearmanr

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SEED=42
DATA=str(ROOT / "data" / "UF_data.csv")
OUT=str(ROOT / "results")
INPUTS=['glu','mlss','air','fm','cn','hrt','srt']; OUTPUTS=['tmp','flow','lv']
df=pd.read_csv(DATA); df.columns=[c.strip().lstrip('﻿') for c in df.columns]
n=len(df); cut=int(0.7*n); X=df[INPUTS].values
mu,sd=X.mean(0),X.std(0); Xz=(X-mu)/sd

# ---------------------------------------------- Part 1
P=pd.read_csv(f"{OUT}/F_conditional_rate_profiles.csv").dropna(subset=['model_rate','real_rate'])
r,p = pearsonr(P.model_rate,P.real_rate)
rho,_ = spearmanr(P.model_rate,P.real_rate)
bias = (P.model_rate-P.real_rate).mean()
mae  = (P.model_rate-P.real_rate).abs().mean()
print("="*88); print("1. MODEL FEASIBILITY RATE vs PLANT-MEASURED ATTAINMENT RATE"); print("="*88)
print(f"bins compared           : {len(P)}")
print(f"Pearson r               : {r:.4f}  (p = {p:.2e})")
print(f"Spearman rho            : {rho:.4f}")
print(f"mean bias (model-real)  : {bias:+.2f} percentage points")
print(f"mean absolute deviation : {mae:.2f} percentage points")
json.dump({'n_bins':int(len(P)),'pearson_r':float(r),'p':float(p),'spearman':float(rho),
           'bias_pp':float(bias),'mad_pp':float(mae)},
          open(f"{OUT}/G_rate_agreement.json","w"),indent=2)
per_feat=[]
for f in INPUTS:
    s=P[P.feature==f]
    if len(s)>=4:
        per_feat.append({'feature':f,'n_bins':len(s),
                         'pearson_r':round(float(pearsonr(s.model_rate,s.real_rate)[0]),3),
                         'mean_bias_pp':round(float((s.model_rate-s.real_rate).mean()),2)})
pf=pd.DataFrame(per_feat); pf.to_csv(f"{OUT}/G_rate_agreement_per_feature.csv",index=False)
print(); print(pf.to_string(index=False))

# ---------------------------------------------- Part 2
print("\n"+"="*88); print("2. FEATURE-IMPORTANCE RANK STABILITY ACROSS SPLIT SCHEMES"); print("="*88)
def blocked(bs,seed=1000,frac=0.7):
    nb=int(np.ceil(n/bs)); B=[np.arange(i*bs,min((i+1)*bs,n)) for i in range(nb)]
    o=np.random.RandomState(seed).permutation(nb); k=int(frac*nb)
    return np.sort(np.concatenate([B[i] for i in o[:k]])), np.sort(np.concatenate([B[i] for i in o[k:]]))
rs=np.random.RandomState(SEED); perm=rs.permutation(n)
SCH={'random':(np.sort(perm[:cut]),np.sort(perm[cut:])),
     'blocked_24h':blocked(24),'blocked_168h':blocked(168),
     'chronological':(np.arange(cut),np.arange(cut,n))}
recs=[]
for t in OUTPUTS:
    for tag,(tr,te) in SCH.items():
        m_=X[tr].mean(0); s_=X[tr].std(0); s_[s_==0]=1
        Xtr,Xte=(X[tr]-m_)/s_,(X[te]-m_)/s_
        y=df[t].values; my,sy=y[tr].mean(),y[tr].std() or 1
        mdl=ExtraTreesRegressor(n_estimators=100,random_state=SEED,n_jobs=-1)
        mdl.fit(Xtr,(y[tr]-my)/sy)
        r2=r2_score(y[te],mdl.predict(Xte)*sy+my)
        pi=permutation_importance(mdl,Xte,(y[te]-my)/sy,n_repeats=5,
                                  random_state=0,n_jobs=-1).importances_mean
        for rank,j in enumerate(np.argsort(-pi),1):
            recs.append({'target':t,'scheme':tag,'test_R2':round(r2,4),
                         'feature':INPUTS[j],'importance':round(float(pi[j]),4),'rank':rank})
I=pd.DataFrame(recs); I.to_csv(f"{OUT}/G_importance_rank_stability.csv",index=False)
for t in OUTPUTS:
    print(f"\n--- {t} : rank by scheme (permutation importance on the held-out fold) ---")
    piv=I[I.target==t].pivot(index='feature',columns='scheme',values='rank')
    piv=piv[['random','blocked_24h','blocked_168h','chronological']].sort_values('random')
    print(piv.to_string())
    print("   test R2:",{k:round(v,3) for k,v in I[I.target==t].groupby('scheme')['test_R2'].first().items()})
rc=[]
for t in OUTPUTS:
    base=I[(I.target==t)&(I.scheme=='random')].set_index('feature')['rank']
    for tag in ['blocked_24h','blocked_168h','chronological']:
        o=I[(I.target==t)&(I.scheme==tag)].set_index('feature')['rank']
        rc.append({'target':t,'scheme':tag,
                   'spearman_vs_random':round(float(spearmanr(base.loc[INPUTS],o.loc[INPUTS]).statistic),3),
                   'same_top_feature':bool(base.idxmin()==o.idxmin())})
RC=pd.DataFrame(rc); RC.to_csv(f"{OUT}/G_rank_correlation.csv",index=False)
print("\nRank agreement with the random-split ranking:")
print(RC.to_string(index=False))

# ---------------------------------------------- Part 3
print("\n"+"="*88); print("3. CONSISTENCY CHECKS"); print("="*88)
r_af=pearsonr(df.air,df.flow); print(f"Pearson r(air, flow) on the raw record : {r_af[0]:.4f} (p={r_af[1]:.2e})")
try:
    Fe=pd.read_csv(f"{OUT}/E_feasible_points_manifold_sample.csv")
    rr=pearsonr(Fe.air,Fe.pred_flow); print(f"Pearson r(air, predicted flow) in feasible region : {rr[0]:.4f}")
except Exception as e: print("feasible file:",e)
corr=df[INPUTS+OUTPUTS].corr()
pairs=[]
for i,a in enumerate(INPUTS+OUTPUTS):
    for b in (INPUTS+OUTPUTS)[i+1:]:
        pairs.append({'pair':f"{a}-{b}",'r':round(float(corr.loc[a,b]),4)})
pp=pd.DataFrame(pairs).reindex(pd.DataFrame(pairs).r.abs().sort_values(ascending=False).index)
pp.to_csv(f"{OUT}/G_all_pairwise_correlations.csv",index=False)
print("\nStrongest linear relationships in the raw dataset:")
print(pp.head(10).to_string(index=False))
TREE=['Decision Tree','Random Forest','Extra Trees','Bagging','AdaBoost','Gradient Boosting',
      'Hist Gradient Boosting','XGBoost','LightGBM']
NONTREE=['Linear Regression','Ridge Regression','Lasso Regression','ElasticNet','KNN','SVR','MLP Neural Network']
print(f"\nTreeSHAP-eligible models  : {len(TREE)}  {TREE}")
print(f"KernelSHAP models         : {len(NONTREE)}  {NONTREE}")
print(f"Total                     : {len(TREE)+len(NONTREE)}")
json.dump({'tree':TREE,'nontree':NONTREE,'n_tree':len(TREE),'n_nontree':len(NONTREE)},
          open(f"{OUT}/G_shap_model_split.json","w"),indent=2)
print(f"\nOutputs -> {OUT}")
