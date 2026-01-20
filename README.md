# high-energy-denoiser

BDT-based denoiser for HEP data using all variables from `ImpactParameterDataset.var_names`.
Trains on MC (signal) vs experimental sideband (background) and saves the model to `models/`.

## Systematic variable testing
Run per-variable BDTs and save ROC/significance plots + summary CSV:

```bash
nix-shell --run "python -m src.main --systematic-test --systematic-output outputs/systematic"
```

Outputs:
- `outputs/systematic/systematic_summary.csv`
- `outputs/systematic/roc_<var>.png`
- `outputs/systematic/significance_<var>.png`
- `outputs/systematic/auc_by_variable.png`
- `outputs/systematic/significance_by_variable.png`


