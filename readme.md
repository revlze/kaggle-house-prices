# House Prices - Advanced Regression Techniques
Need to predict sales prices, I tried many models, like catboost, xgboost, linear models, svr.

## Final result

Stack SVR + CatBoost, `pruned_ordinal_agg`, mean of 5 fold models: public LB 0.12332.

## Solutions
| Approach | CV  | CV STD | LB  | Date |
| -------- | --- | ------ | --- | ---- |
| catboost (default), pruned_ordinal, StratifiedKFold(5) | 0.11312 | 0.00549 | 0.12793 | 2026-11-11 |
| catboost (optuna 30 trials), pruned_ordinal, StratifiedKFold(5) | 0.11305 | 0.00419 | 0.12794 | 2026-11-12 |
| svr (tuned 50 trials), pruned_ordinal, StratifiedKFold(5) | 0.10975 | 0.00298 | 0.12625 | 2026-11-12 |
| stack svr + catboost, pruned_ordinal, StratifiedKFold(5) | 0.10678 | 0.00354 | 0.12422 | 2026-11-13 |
| svr (default), pruned_ordinal_agg, StratifiedKFold(5) | 0.10902 | 0.00299 | 0.12546 | 2026-11-13 |
| svr (tuned 50 trials), pruned_ordinal_agg, StratifiedKFold(5) | 0.10901 | 0.00323 | 0.12765 | 2026-11-13 |
| stack svr + catboost, pruned_ordinal_agg, StratifiedKFold(5) | 0.10691 | 0.00347 | 0.12354 | 2026-11-13 |
| same stack, 1 fold model | 0.10691 | 0.00347 | 0.12447 | 2026-11-13 |
| same stack, mean of 5 fold models | 0.10691 | 0.00347 | 0.12332 | 2026-11-13 |

Other models, `pruned_ordinal`, StratifiedKFold(5), CV only:

| Model | CV | CV STD |
| ----- | -- | ------ |
| linear regression (baseline) | 0.12407 | 0.00263 |
| ridge (tuned) | 0.11318 | 0.00490 |
| elasticnet (tuned) | 0.11823 | 0.00452 |
| knn (tuned) | 0.16114 | 0.00687 |
| decision tree (tuned) | 0.18159 | 0.00548 |
| random forest (tuned) | 0.13225 | 0.00304 |
| xgboost (tuned) | 0.11362 | 0.00546 |
| mlp (128-64, dropout 0.1, AdamW, cosine) | 0.12693 | 0.00221 |

EDA: `notebooks/01_edaa.ipynb` (profiling reports in `notebooks/htmls`).

## Setup

Put the [Kaggle data](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) into `data/`.

### Run

```bash
uv sync
uv run run.py model=stack_svr_catboost feature_set=pruned_ordinal_agg name_exp=final_stack predict_with=folds
```