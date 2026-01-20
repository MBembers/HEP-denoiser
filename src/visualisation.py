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
    zoom_percentiles: tuple[float, float] = (0.5, 99.5),
):
    """Plot histogram comparing experimental vs MC data for a variable."""

    plt.style.use(mplhep.style.ATLAS)
    exp_values = _sanitize_series(exp_df[var])
    mc_values = _sanitize_series(mc_df[var])
    all_values = [exp_values, mc_values]
    plt.hist(exp_values, bins=bins, alpha=0.5, label="Experimental", color="blue", density=True)
    plt.hist(mc_values, bins=bins, alpha=0.5, label="MC", color="orange", density=True)
    if filtered_df is not None:
        filt_values = _sanitize_series(filtered_df[var])
        all_values.append(filt_values)
        plt.hist(filt_values, bins=bins, alpha=0.5, label="Filtered", color="green", density=True)
    if all_values:
        combined = np.concatenate([vals for vals in all_values if vals.size])
        if combined.size:
            low, high = np.percentile(combined, zoom_percentiles)
            if np.isfinite(low) and np.isfinite(high) and low < high:
                plt.xlim(low, high)
    plt.xlabel(var)
    plt.ylabel("Density")
    plt.title(f"Histogram of {var}")
    plt.legend()
    _plot_save(output_dir, f"{var}_comparison.png")


def plot_var_triptych(
    var: str,
    exp_df: pd.DataFrame,
    mc_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    *,
    bins: int = 70,
    output_dir: Optional[str] = None,
    zoom_percentiles: tuple[float, float] = (0.5, 99.5),
) -> None:
    """Plot MC/EXP, MC/Filtered, EXP/Filtered as three subplots with outlines."""
    exp_values = _sanitize_series(exp_df[var])
    mc_values = _sanitize_series(mc_df[var])
    filt_values = _sanitize_series(filtered_df[var])
    combined = np.concatenate([exp_values, mc_values, filt_values]) if exp_values.size or mc_values.size else filt_values

    low, high = np.percentile(combined, zoom_percentiles) if combined.size else (None, None)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    plt.style.use(mplhep.style.ATLAS)

    def _hist(ax, a, b, label_a, label_b, color_a, color_b):
        ax.hist(a, bins=bins, histtype="step", density=True, label=label_a, color=color_a, linewidth=1.5)
        ax.hist(b, bins=bins, histtype="step", density=True, label=label_b, color=color_b, linewidth=1.5)
        ax.set_title(f"{label_a} vs {label_b}")
        ax.set_xlabel(var)
        ax.legend(fontsize=8)
        if low is not None and high is not None and np.isfinite(low) and np.isfinite(high) and low < high:
            ax.set_xlim(low, high)

    _hist(axes[0], mc_values, exp_values, "MC", "EXP", "orange", "blue")
    _hist(axes[1], mc_values, filt_values, "MC", "Filtered", "orange", "green")
    _hist(axes[2], exp_values, filt_values, "EXP", "Filtered", "blue", "green")

    axes[0].set_ylabel("Density")
    fig.suptitle(f"{var} distributions")
    fig.tight_layout()
    _plot_save(output_dir, f"{var}_triptych.png")


def plot_var_combined(
    var: str,
    exp_df: pd.DataFrame,
    mc_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    *,
    bins: int = 70,
    output_dir: Optional[str] = None,
    zoom_percentiles: tuple[float, float] = (0.5, 99.5),
) -> None:
    """Plot MC/EXP/Filtered together with outline-only histograms."""
    exp_values = _sanitize_series(exp_df[var])
    mc_values = _sanitize_series(mc_df[var])
    filt_values = _sanitize_series(filtered_df[var])
    combined = np.concatenate([exp_values, mc_values, filt_values]) if exp_values.size or mc_values.size else filt_values

    low, high = np.percentile(combined, zoom_percentiles) if combined.size else (None, None)

    plt.figure(figsize=(6, 4))
    plt.style.use(mplhep.style.ATLAS)
    plt.hist(mc_values, bins=bins, histtype="step", density=True, label="MC", color="green", linewidth=1.5)
    plt.hist(exp_values, bins=bins, histtype="step", density=True, label="EXP", color="gold", linewidth=1.5)
    plt.hist(filt_values, bins=bins, histtype="step", density=True, label="Filtered", color="red", linewidth=1.5)
    if low is not None and high is not None and np.isfinite(low) and np.isfinite(high) and low < high:
        plt.xlim(low, high)
    plt.xlabel(var)
    plt.ylabel("Density")
    plt.title(f"{var} distributions")
    plt.legend()
    _plot_save(output_dir, f"{var}_combined.png")


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


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    *,
    output_dir: Optional[str] = None,
    filename: str = "roc_curve.png",
) -> None:
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Random guess")
    plt.plot(fpr, tpr, label=f"ROC (AUC={auc_score:.2f})")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.gca().set_aspect("equal", adjustable="box")
    _plot_save(output_dir, filename)


def plot_significance_curve(
    thresholds: np.ndarray,
    metric: np.ndarray,
    *,
    output_dir: Optional[str] = None,
    filename: str = "significance_curve.png",
) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(thresholds, metric, label="$S/\sqrt{S+B}$")
    plt.xlabel("BDT cut value")
    plt.ylabel("$S/\sqrt{S+B}$")
    plt.xlim(0.0, 1.0)
    plt.legend(loc="best")
    _plot_save(output_dir, filename)


def plot_metric_bar(
    summary_df: pd.DataFrame,
    metric: str,
    *,
    output_dir: Optional[str] = None,
    filename: str = "metric_bar.png",
    top_n: int = 20,
) -> None:
    if metric not in summary_df.columns or summary_df.empty:
        return
    ordered = summary_df.sort_values(metric, ascending=False).head(top_n)
    plt.figure(figsize=(10, 4))
    plt.bar(ordered["feature_set"], ordered[metric], color="steelblue")
    plt.xticks(rotation=90, fontsize=8)
    plt.ylabel(metric)
    plt.title(f"{metric} by variable")
    plt.tight_layout()
    _plot_save(output_dir, filename)


def plot_metric_scatter(
    summary_df: pd.DataFrame,
    *,
    x_metric: str,
    y_metric: str,
    output_dir: Optional[str] = None,
    filename: str = "metric_scatter.png",
) -> None:
    if summary_df.empty or x_metric not in summary_df.columns or y_metric not in summary_df.columns:
        return
    plt.figure(figsize=(6, 4))
    plt.scatter(summary_df[x_metric], summary_df[y_metric], alpha=0.8, color="darkred")
    plt.xlabel(x_metric)
    plt.ylabel(y_metric)
    plt.title(f"{y_metric} vs {x_metric}")
    plt.tight_layout()
    _plot_save(output_dir, filename)


def plot_roc_overlay(
    roc_curves: list[tuple[str, np.ndarray, np.ndarray, float]],
    *,
    output_dir: Optional[str] = None,
    filename: str = "roc_overlay.png",
) -> None:
    if not roc_curves:
        return
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Random guess")
    for label, fpr, tpr, auc_score in roc_curves:
        plt.plot(fpr, tpr, label=f"{label} (AUC={auc_score:.2f})")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", fontsize=8)
    plt.gca().set_aspect("equal", adjustable="box")
    _plot_save(output_dir, filename)

