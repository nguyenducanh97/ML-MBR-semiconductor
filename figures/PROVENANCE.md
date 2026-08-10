# Which script produces which figure

| Figure | Content | Produced by |
| --- | --- | --- |
| Graphical abstract | summary panel | `code/22_graphical_abstract.py` |
| 1 | framework workflow | `code/23_workflow_diagram.py` |
| 2 | time series and distributions, before and after standardisation | `code/00_modelling_pipeline.ipynb`, stage 4 |
| 3 | correlation matrices before and after standardisation | `code/00_modelling_pipeline.ipynb`, stage 4 |
| 4 | R² heatmap over 16 models and predicted against observed | `code/00_modelling_pipeline.ipynb`, stage 6.1 |
| 5 | transfer to the parallel B stream | `code/00_modelling_pipeline.ipynb`, stage 8 |
| 6 | SHAP importance heatmaps and beeswarm across 16 models | `code/00_modelling_pipeline.ipynb`, stage 7 |
| 7 | feasible range and measured attainment per input | `code/21_figures_feasible_region.py` |
| 8 | pairwise conditional feasibility rate | `code/21_figures_feasible_region.py` |
| 9 | predicted output response across the feasible region | `code/21_figures_feasible_region.py` |
| 10 | model rate against measured attainment | `code/20_figures_validation.py` |
| 11 | recommended window and its out-of-time test | `code/20_figures_validation.py` |
| S1–S3 | predicted against observed for all 16 models, one per target | `code/00_modelling_pipeline.ipynb`, stage 6.1 |
| S4–S6 | R², MAE and RMSE consistency across 20 random splits | `code/00_modelling_pipeline.ipynb`, stage 6.2 |
| S7 | time series of the B stream | `code/00_modelling_pipeline.ipynb`, stage 8 |
| S8 | TreeSHAP against KernelSHAP, and feature interactions | `code/00_modelling_pipeline.ipynb`, stage 7 |
| S9–S11 | SHAP dependence plots for Extra Trees, one per target | `code/00_modelling_pipeline.ipynb`, stage 7 |
| S12 | feasible input ranges | `code/00_modelling_pipeline.ipynb`, stage 10 |
| S13 | predicted output distributions within the feasible region | `code/00_modelling_pipeline.ipynb`, stage 10 |
| S14–S16 | SHAP beeswarm and dependence within the feasible region | `code/00_modelling_pipeline.ipynb`, stage 10 |
| S17 | temporal structure, error scale and refit schedule | `code/20_figures_validation.py` |
| S18 | per-lever transfer to the held-out period | `code/20_figures_validation.py` |

The notebook covers data loading through to the first feasible-region analysis.
The numbered scripts cover the validation designs, the manifold-constrained
resampling and the operating-window derivation, and they regenerate the figures
that replaced their earlier counterparts.
