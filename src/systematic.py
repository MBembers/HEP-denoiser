from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .data_loader import ImpactParameterDataset
from .evaluate import compute_roc, compute_significance_curve
from .model import BDTDenoiser
from .visualisation import (
	plot_metric_bar,
	plot_metric_scatter,
	plot_roc_curve,
	plot_roc_overlay,
	plot_significance_curve,
)


@dataclass
class SystematicResult:
	feature_set: str
	n_features: int
	auc: float
	best_significance: float
	best_threshold: float


def _safe_name(name: str) -> str:
	return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name)


def run_systematic_variable_tests(
	*,
	background_window: tuple[float, float] = (5700.0, 5900.0),
	output_dir: Path = Path("outputs/systematic"),
	include_all: bool = True,
	make_plots: bool = True,
	bdt_params: Optional[dict] = None,
) -> pd.DataFrame:
	"""Train per-variable BDTs and evaluate ROC/AUC and significance curves."""
	output_dir.mkdir(parents=True, exist_ok=True)
	bdt_params = bdt_params or {}

	dataset = ImpactParameterDataset()
	dataset.load_experimental()
	dataset.load_mc()

	exp_df = dataset.to_dataframe("exp")
	mc_df = dataset.to_dataframe("mc")
	low, high = background_window
	bkg_df = exp_df.query(f"~({low} < Xb_M < {high})").copy()

	feature_sets: list[tuple[str, list[str]]] = [(var, [var]) for var in dataset.var_names]
	if include_all:
		feature_sets.append(("ALL", dataset.var_names))

	results: list[SystematicResult] = []
	roc_cache: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
	for label, features in feature_sets:
		denoiser = BDTDenoiser(features=tuple(features))
		denoiser.fit(mc_df, bkg_df, **bdt_params)
		mc_scores = denoiser.predict_proba(mc_df)
		bkg_scores = denoiser.predict_proba(bkg_df)

		y_true = np.concatenate([np.ones_like(mc_scores), np.zeros_like(bkg_scores)])
		y_score = np.concatenate([mc_scores, bkg_scores])
		roc = compute_roc(y_true, y_score)
		roc_cache[label] = (roc.fpr, roc.tpr, roc.auc_score)
		significance_df = compute_significance_curve(
			y_true=y_true,
			y_score=y_score,
			n_sig=float(len(mc_df)),
			n_bkg=float(len(bkg_df)),
		)
		best_idx = int(np.argmax(significance_df["significance"].to_numpy()))
		best_sig = float(significance_df["significance"].iloc[best_idx])
		best_thr = float(significance_df["threshold"].iloc[best_idx])

		results.append(
			SystematicResult(
				feature_set=label,
				n_features=len(features),
				auc=roc.auc_score,
				best_significance=best_sig,
				best_threshold=best_thr,
			)
		)

		if make_plots:
			stem = _safe_name(label)
			plot_roc_curve(roc.fpr, roc.tpr, roc.auc_score, output_dir=str(output_dir), filename=f"roc_{stem}.png")
			plot_significance_curve(
				significance_df["threshold"].to_numpy(),
				significance_df["significance"].to_numpy(),
				output_dir=str(output_dir),
				filename=f"significance_{stem}.png",
			)

	summary_df = pd.DataFrame([r.__dict__ for r in results]).sort_values("auc", ascending=False)
	summary_path = output_dir / "systematic_summary.csv"
	summary_df.to_csv(summary_path, index=False)

	if make_plots:
		plot_metric_bar(summary_df, "auc", output_dir=str(output_dir), filename="auc_by_variable.png")
		plot_metric_bar(
			summary_df,
			"best_significance",
			output_dir=str(output_dir),
			filename="significance_by_variable.png",
		)
		plot_metric_bar(
			summary_df,
			"auc",
			output_dir=str(output_dir),
			filename="auc_top10.png",
			top_n=10,
		)
		plot_metric_bar(
			summary_df,
			"best_significance",
			output_dir=str(output_dir),
			filename="significance_top10.png",
			top_n=10,
		)
		plot_metric_scatter(
			summary_df,
			x_metric="auc",
			y_metric="best_significance",
			output_dir=str(output_dir),
			filename="auc_vs_significance.png",
		)
		# ROC overlay for top-5 AUC variables
		top_labels = summary_df.head(5)["feature_set"].tolist()
		roc_curves = [
			(label, *roc_cache[label])
			for label in top_labels
			if label in roc_cache
		]
		plot_roc_overlay(
			roc_curves,
			output_dir=str(output_dir),
			filename="roc_overlay_top5.png",
		)

	print("\nSystematic variable testing summary")
	print("-----------------------------------")
	print(summary_df.head(15).to_string(index=False))
	best_auc = summary_df.iloc[0]
	print(
		f"\nBest AUC: {best_auc['feature_set']} (AUC={best_auc['auc']:.3f}, "
		f"best S/sqrt(S+B)={best_auc['best_significance']:.3f})"
	)
	low_auc = summary_df.tail(5)
	print("\nLowest-AUC variables (candidates to drop):")
	print(low_auc[["feature_set", "auc", "best_significance"]].to_string(index=False))

	return summary_df
