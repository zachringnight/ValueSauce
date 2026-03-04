# ValueSauce

A lightweight **basketball value model** for estimating team strength from box-score style features.

## Model highlights

- Ridge-regularized linear regression for stability under correlated basketball metrics.
- Automatic feature standardization to prevent scale dominance (pace vs. rates vs. percentages).
- Optional sample weighting for emphasizing high-confidence or high-leverage games.
- Built-in diagnostics (`score_r2`, `score_rmse`) and sorted coefficient importance.
- Cross-validated ridge tuning via `tune_ridge_alpha`.

## Run tests

```bash
pytest -q
```
