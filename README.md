# high-energy-denoiser

BDT-based denoiser for HEP data using all variables from `ImpactParameterDataset.var_names`.
Trains on MC (signal) vs experimental sideband (background) and saves the model to `models/`.

## Quick start
Run inside nix-shell so the ROOT/UPROOT environment is available:

```bash
nix-shell --run "python -m src.main --plot --plot-dir outputs/plots"
```

## Options
- `--threshold`: BDT score threshold for filtering (default 0.95)
- `--background-window`: Sideband mass window excluded from background (default `5700,5900`)
- `--model-dir`: Directory for saved model artifacts (default `models`)
- `--plot-dir`: Directory where plots are saved


## Notes
Add decay time features: `Xb_TAU`, `Xc_TAU`, `Xb_TAUERR`, `Xc_TAUERR`, `Xb_TAUCHI2`.



