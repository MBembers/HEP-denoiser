# high-energy-denoiser

Gaussian denoiser that learns the MC feature distribution and filters experimental events
based on Mahalanobis distance. Includes bootstrap-based benchmarking and KS diagnostics.

## What it does
- Trains a simple NN classifier (PyTorch MLP) to distinguish MC vs EXP.
- Scores experimental events by MC-likeness probability.
- Filters events above an MC probability threshold (denoised subset).
- Benchmarks with bootstrap mean CIs and KS statistics per feature.

## Quick start
Run inside nix-shell so the ROOT/UPROOT environment is available:

```bash
nix-shell --run "python -m src.main --mode classifier --retain-quantile 0.95"
```

To save the filtered experimental data:

```bash
nix-shell --run "python -m src.main --mode classifier --retain-quantile 0.95 --save-filtered filtered_exp.csv"

# Optional: force device
nix-shell --run "python -m src.main --mode classifier --device cpu"

# Optional: save plots
nix-shell --run "python -m src.main --mode classifier --plot-dir outputs/plots"

# Optional: target a higher retention rate (keeps more EXP)
nix-shell --run "python -m src.main --mode classifier --target-retention 0.5"

# Optional: sweep quantiles and save retention vs KS curve
nix-shell --run "python -m src.main --mode classifier --plot-dir outputs/plots --sweep-quantiles 0.80,0.85,0.90,0.95,0.98"
```

Gaussian baseline (distance-based):

```bash
nix-shell --run "python -m src.main --mode gaussian --quantile 0.95"
```

## Notes / future ideas
- Add decay time features: `Xb_TAU`, `Xc_TAU`, `Xb_TAUERR`, `Xc_TAUERR`, `Xb_TAUCHI2`.
- Extend benchmarks with additional distance metrics or classifier-based validation.



