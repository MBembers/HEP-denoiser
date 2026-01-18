# src/utils.py
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Union
import numpy as np
import uproot
# Change .dataloader to .data_loader
from .data_loader import DataLoader

# ---- ROOT files ----
BASE_DIR = Path(__file__).resolve().parent.parent
FILES = [
    BASE_DIR / "data" / "data_Xib2XicPi_2016_MU.addVar.wMVA.root",
    BASE_DIR / "data" / "MC_Xib2XicPi_2016MC_MU.pid.addVar.wMVA.root"
]

TREES = ["mytree"] 


def bootstrap_statistic(
    data: Union[Sequence[float], np.ndarray],
    statistic: Callable[[np.ndarray], float],
    *,
    n_resamples: int = 2000,
    sample_size: Optional[int] = None,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None,
    return_distribution: bool = False,
) -> Dict[str, float]:
    """Compute a bootstrap estimate and confidence interval for a statistic.

    Args:
        data: 1D array-like sample.
        statistic: Callable applied to each bootstrap resample.
        n_resamples: Number of bootstrap resamples.
        sample_size: Size of each resample (defaults to len(data)).
        confidence_level: Confidence level for percentile interval.
        random_state: Seed for reproducibility.
        return_distribution: Include the full bootstrap distribution if True.

    Returns:
        Dictionary with the point estimate and confidence interval.
    """
    values = np.asarray(data, dtype=float)
    values = values[~np.isnan(values)]

    if values.ndim != 1:
        raise ValueError("data must be 1D")
    if values.size == 0:
        raise ValueError("data must contain at least one finite value")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError("confidence_level must be between 0 and 1")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")

    rng = np.random.default_rng(random_state)
    sample_size = sample_size or values.size

    estimates = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        resample = rng.choice(values, size=sample_size, replace=True)
        estimates[i] = statistic(resample)

    alpha = (1.0 - confidence_level) / 2.0
    lower = np.quantile(estimates, alpha)
    upper = np.quantile(estimates, 1.0 - alpha)
    point = statistic(values)

    result: Dict[str, float] = {
        "estimate": float(point),
        "ci_low": float(lower),
        "ci_high": float(upper),
        "confidence_level": float(confidence_level),
        "n_resamples": int(n_resamples),
        "sample_size": int(sample_size),
    }

    if return_distribution:
        result["distribution"] = estimates

    return result


def print_tree_variables(file_path, tree_name):
    """Print all variables/branches inside a ROOT tree."""
    loader = DataLoader(file_path, tree_name)
    tree = loader.get_tree()
    print(f"\nFile: {file_path}")
    print(f"Tree: {tree_name}")
    print("Branches / Variables:")
    for branch in tree.keys():
        print(f" - {branch}")


if __name__ == "__main__":
    for f in FILES:
        for t in TREES:
            try:
                print_tree_variables(f, t)
            except Exception as e:
                print(f"Could not access {t} in {f}: {e}")

