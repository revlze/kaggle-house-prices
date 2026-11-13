# House Prices - Advanced Regression Techniques
Need to predict sales prices, I tried many models, like catboost, xgboost, linear models, svr.

## Final result

Stack SVR + CatBoost, `pruned_ordinal_agg`: public LB 0.12354.

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

## Setup

Put the [Kaggle data](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) into `data/`.

### Run

```bash
uv sync
uv run run.py model=stack_svr_catboost feature_set=pruned_ordinal_agg name_exp=final_stack
```