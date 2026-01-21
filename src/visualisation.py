import pandas as pd
import matplotlib.pyplot as plt
import mplhep

# only place for plotting functions and only plotting functions
def plot_var_from_df(var: str, df: pd.DataFrame, bins: int = 70):
    """Plot histogram for a single DataFrame variable."""
    plt.figure(figsize=(6,6))
    plt.style.use(mplhep.style.ATLAS)
    plt.hist(df[var], bins=bins, alpha=0.7, label="Data", color="green", density=True)
    plt.xlabel(var)
    plt.ylabel("Density")
    plt.title(f"Histogram of {var}")
    plt.legend()
    plt.savefig(f"{var}_histogram.png")
    plt.close()


def plot_var_comparison(var: str, exp_df: pd.DataFrame, mc_df: pd.DataFrame, bins: int = 70):
    """Plot histogram comparing experimental vs MC data for a variable."""
    plt.figure(figsize=(6,6))
    plt.style.use(mplhep.style.ATLAS)
    plt.hist(exp_df[var], bins=bins, alpha=0.5, label="Experimental", color="blue", density=True)
    plt.hist(mc_df[var], bins=bins, alpha=0.5, label="MC", color="orange", density=True)
    plt.xlabel(var)
    plt.ylabel("Density")
    plt.title(f"Histogram of {var}")
    plt.legend()
    plt.savefig(f"{var}_comparison.png")
    plt.close()

def add_var_to_plot(var: str, df: pd.DataFrame, bins: int = 70, **kwargs):
    """Add variable histogram to existing plot."""
    plt.hist(df[var], bins=bins, alpha=0.5, density=True, histtype="step", **kwargs)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title("Overlayed Histograms")

