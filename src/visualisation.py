from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep

# only place for plotting functions and only plotting functions
def _plot_save(path: Optional[str], filename: str) -> None:
    if path is None:
        plt.savefig(filename)
        plt.close()
        return
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / filename)
    plt.close()


def plot_var_from_df(var: str, df: pd.DataFrame, bins: int = 70, output_dir: Optional[str] = None):
    """Plot histogram for a single DataFrame variable."""
    plt.style.use(mplhep.style.ATLAS)
    values = _sanitize_series(df[var])
    plt.hist(values, bins=bins, alpha=0.7, label="Data", color="green", density=True)
    plt.xlabel(var)
    plt.ylabel("Density")
    plt.title(f"Histogram of {var}")
    plt.legend()
    _plot_save(output_dir, f"{var}_histogram.png")


def plot_var_comparison(
    var: str,
    exp_df: pd.DataFrame,
    mc_df: pd.DataFrame,
    bins: int = 70,
    output_dir: Optional[str] = None,
    filtered_df: Optional[pd.DataFrame] = None,
):
    """Plot histogram comparing experimental vs MC data for a variable."""

    plt.style.use(mplhep.style.ATLAS)
    exp_values = _sanitize_series(exp_df[var])
    mc_values = _sanitize_series(mc_df[var])
    plt.hist(exp_values, bins=bins, alpha=0.5, label="Experimental", color="blue", density=True)
    plt.hist(mc_values, bins=bins, alpha=0.5, label="MC", color="orange", density=True)
    if filtered_df is not None:
        filt_values = _sanitize_series(filtered_df[var])
        plt.hist(filt_values, bins=bins, alpha=0.5, label="Filtered", color="green", density=True)
    plt.xlabel(var)
    plt.ylabel("Density")
    plt.title(f"Histogram of {var}")
    plt.legend()
    _plot_save(output_dir, f"{var}_comparison.png")


def _sanitize_series(series: pd.Series) -> np.ndarray:
    values = series.to_numpy()
    if values.dtype == object:
        flat: list[float] = []
        for item in values:
            if item is None:
                continue
            if np.isscalar(item):
                flat.append(float(item))
            else:
                flat.extend(np.asarray(item, dtype=float).ravel().tolist())
        values = np.asarray(flat, dtype=float)
    else:
        values = values.astype(float)
    values = values[np.isfinite(values)]
    return values

