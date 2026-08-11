# Interpretable machine learning for a full-scale semiconductor-wastewater MBR
Yujae Jeon1, Duc Anh Nguyen2, Kim Anh Nguyen Thi2, Quoc Thai Nong2, Am Jang2*

1Department of Semiconductor Display Engineering, Sungkyunkwan University (SKKU), (16419) 2066, Seobu-ro, Jangan-gu, Suwon, Gyeonggi-do, Republic of Korea

2Department of Global Smart City, Sungkyunkwan University (SKKU), (16419) 2066, Seobu-ro, Jangan-gu, Suwon, Gyeonggi-do, Republic of Korea

DOI: 

*Corresponding author: Tel.: +82-31-290-7526; Fax: +82-31-290-7549; Email: amjang@skku.edu

Authors' emails: Yujae Jeon (junyujae@naver.com), Duc Anh Nguyen (nguyenducanh@g.skku.edu), Kim Anh Nguyen Thi (toikimanh@gmail.com), Quoc Thai Nong (thainq@g.skku.edu).

Code, derived results and figures for a study of a full-scale submerged membrane
bioreactor treating semiconductor fabrication wastewater. Sixteen regression
algorithms are benchmarked for transmembrane pressure, permeate flow and
membrane tank water level; SHAP is applied across all of them; and the feasible
operating region is mapped by resampling constrained to the plant's real joint
operating manifold.

## Data availability

The plant's raw SCADA record is owned by the operating company and cannot be
redistributed.

`results/`: every derived quantity behind the figures and tables: model
performance under each validation design, SHAP importance, conditional
feasibility rates, the sampled operating manifold and the operating-window
statistics. These are aggregates or model-generated states, not plant records.


## Layout

```
code/     the modelling notebook and the analysis scripts, in execution order
figures/  figures as they appear in the paper (main/ and supplementary/)
results/  derived outputs, grouped by analysis
```

### code

| script | purpose |
| --- | --- |
| `00_modelling_pipeline.ipynb`| covers data loading and preprocessing, the benchmark of all sixteen algorithms, the multi-split robustness analysis, SHAP interpretability, validation on the parallel B stream and the first feasible-region analysis, and it produces Figures 2 to 6 and Figures S1 to S16 |
| `01_validation_designs.py` | all 16 models under random, chronological, purged and blocked partitions |
| `02_block_size_sweep.py` | autocorrelation, accuracy against splitting block length, covariate coverage |
| `03_window_audit.py` | model-free audit of an operating window against the logged record |
| `04_dose_response.py` | decile dose-response profiles and joint support |
| `05_manifold_sampling.py` | local-covariance kernel resampling of the operating manifold |
| `06_conditional_rate_window.py` | conditional feasibility rate and the per-variable bands |
| `07_rate_agreement.py` | model rate against measured attainment; SHAP rank stability |
| `08_manifold_sample_export.py` | export the full sample and the per-model blocked table |
| `09_band_selection.py` | operability-aware band selection and the out-of-time test |
| `10_band_rounding.py` | round bands to operational precision and recompute |
| `11_tuning_and_feature_trials.py` | hyperparameter and rolling-feature trials under blocked CV |
| `12_error_scale.py` | absolute error alongside R² across block lengths |
| `13_refit_schedule.py` | held-out accuracy against how often the model is refitted |
| `20`–`23` | figures, graphical abstract and workflow diagram |


Notebook outputs are stripped, because several cells printed rows of the raw
operating record.

Scripts resolve their own paths relative to the repository root, so they can be
run from anywhere.

## Requirements

Python 3.10 or later.

```
numpy pandas scipy scikit-learn xgboost lightgbm shap matplotlib pillow
```

```bash
pip install -r requirements.txt
```

## Licence

MIT, see `LICENSE`. Please cite the paper if you use this material.

## Best regards

Thank you very much for your time reading our paper.
