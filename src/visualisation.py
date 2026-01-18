"""Plotting utilities for denoiser diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_dir(output_dir: Optional[str]) -> Optional[Path]:
	if output_dir is None:
		return None
	path = Path(output_dir)
	path.mkdir(parents=True, exist_ok=True)
	return path


def plot_correlation_heatmap(
	df: pd.DataFrame,
	*,
	title: str,
	output_dir: Optional[str] = None,
	filename: str = "correlation_heatmap.png",
) -> None:
	corr = df.corr(numeric_only=True)
	plt.figure(figsize=(10, 8))
	plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
	plt.colorbar(label="Correlation")
	plt.title(title)
	plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=8)
	plt.yticks(range(len(corr.index)), corr.index, fontsize=8)
	plt.tight_layout()

	out_dir = _ensure_dir(output_dir)
	if out_dir:
		plt.savefig(out_dir / filename, dpi=200)
	else:
		plt.show()
	plt.close()


def plot_score_distributions(
	scores_mc: np.ndarray,
	scores_exp: np.ndarray,
	scores_filtered: np.ndarray,
	*,
	title: str,
	output_dir: Optional[str] = None,
	filename: str = "score_distributions.png",
) -> None:
	plt.figure(figsize=(8, 5))
	plt.hist(scores_mc, bins=60, alpha=0.6, label="MC", density=True)
	plt.hist(scores_exp, bins=60, alpha=0.6, label="EXP", density=True)
	plt.hist(scores_filtered, bins=60, alpha=0.6, label="Filtered", density=True)
	plt.legend()
	plt.xlabel("Score / MC-likeness")
	plt.ylabel("Density")
	plt.title(title)
	plt.tight_layout()

	out_dir = _ensure_dir(output_dir)
	if out_dir:
		plt.savefig(out_dir / filename, dpi=200)
	else:
		plt.show()
	plt.close()


def plot_feature_ks_improvement(
	per_feature: dict,
	*,
	output_dir: Optional[str] = None,
	filename: str = "ks_improvement.png",
) -> None:
	features = list(per_feature.keys())
	ks_exp = [per_feature[f]["ks_exp_vs_mc"] for f in features]
	ks_filt = [per_feature[f]["ks_filt_vs_mc"] for f in features]

	x = np.arange(len(features))
	width = 0.35

	plt.figure(figsize=(10, 5))
	plt.bar(x - width / 2, ks_exp, width, label="EXP vs MC")
	plt.bar(x + width / 2, ks_filt, width, label="Filtered vs MC")
	plt.xticks(x, features, rotation=90, fontsize=8)
	plt.ylabel("KS statistic")
	plt.title("Feature KS comparison")
	plt.legend()
	plt.tight_layout()

	out_dir = _ensure_dir(output_dir)
	if out_dir:
		plt.savefig(out_dir / filename, dpi=200)
	else:
		plt.show()
	plt.close()


def plot_feature_histograms(
	df_mc: pd.DataFrame,
	df_exp: pd.DataFrame,
	df_filtered: pd.DataFrame,
	*,
	features: Iterable[str],
	output_dir: Optional[str] = None,
	max_features: int = 8,
) -> None:
	out_dir = _ensure_dir(output_dir)
	for idx, feat in enumerate(list(features)[:max_features]):
		plt.figure(figsize=(6, 4))
		plt.hist(df_mc[feat], bins=60, alpha=0.6, label="MC", density=True)
		plt.hist(df_exp[feat], bins=60, alpha=0.6, label="EXP", density=True)
		plt.hist(df_filtered[feat], bins=60, alpha=0.6, label="Filtered", density=True)
		plt.title(f"{feat} distribution")
		plt.xlabel(feat)
		plt.ylabel("Density")
		plt.legend()
		plt.tight_layout()
		if out_dir:
			plt.savefig(out_dir / f"hist_{idx:02d}_{feat}.png", dpi=200)
		else:
			plt.show()
		plt.close()


def plot_retention_vs_ks(
	sweep_df: pd.DataFrame,
	*,
	output_dir: Optional[str] = None,
	filename: str = "retention_vs_ks.png",
) -> None:
	if sweep_df.empty:
		return
	plt.figure(figsize=(6, 4))
	plt.plot(sweep_df["retention"], sweep_df["avg_ks"], marker="o")
	plt.xlabel("Retention rate")
	plt.ylabel("Average KS vs MC")
	plt.title("Retention vs average KS")
	plt.grid(True, alpha=0.3)
	plt.tight_layout()

	out_dir = _ensure_dir(output_dir)
	if out_dir:
		plt.savefig(out_dir / filename, dpi=200)
	else:
		plt.show()
	plt.close()
