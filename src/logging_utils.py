from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def format_classification_metrics(metrics: Dict[str, float]) -> str:
	return (
		"precision={precision:.3f}, recall={recall:.3f}, accuracy={accuracy:.3f}, "
		"f1={f1:.3f} | tp={tp:.0f}, fp={fp:.0f}, tn={tn:.0f}, fn={fn:.0f}"
	).format(**metrics)


def _ks_statistic(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
	a = np.sort(sample_a)
	b = np.sort(sample_b)
	if a.size == 0 or b.size == 0:
		return float("nan")
	data_all = np.sort(np.concatenate([a, b]))
	cdf_a = np.searchsorted(a, data_all, side="right") / a.size
	cdf_b = np.searchsorted(b, data_all, side="right") / b.size
	return float(np.max(np.abs(cdf_a - cdf_b)))


def summarize_variable_stats(
	var: str,
	mc_df: pd.DataFrame,
	exp_df: pd.DataFrame,
	filtered_df: pd.DataFrame,
) -> Dict[str, float]:
	mc_vals = mc_df[var].to_numpy(dtype=float)
	exp_vals = exp_df[var].to_numpy(dtype=float)
	filt_vals = filtered_df[var].to_numpy(dtype=float)
	ks_exp_mc = _ks_statistic(exp_vals, mc_vals)
	ks_filt_mc = _ks_statistic(filt_vals, mc_vals)
	return {
		"mc_mean": float(np.nanmean(mc_vals)),
		"exp_mean": float(np.nanmean(exp_vals)),
		"filt_mean": float(np.nanmean(filt_vals)),
		"mc_std": float(np.nanstd(mc_vals)),
		"exp_std": float(np.nanstd(exp_vals)),
		"filt_std": float(np.nanstd(filt_vals)),
		"ks_exp_mc": float(ks_exp_mc),
		"ks_filt_mc": float(ks_filt_mc),
	}


def print_variable_table(summary: Dict[str, Dict[str, float]]) -> None:
	header = (
		f"{'var':<22} {'mc_mean':>10} {'exp_mean':>10} {'filt_mean':>10} "
		f"{'mc_std':>10} {'exp_std':>10} {'filt_std':>10} {'ks_exp':>8} {'ks_filt':>8}"
	)
	print("\nPer-variable summary")
	print("-" * len(header))
	print(header)
	for var, stats in summary.items():
		print(
			f"{var:<22} {stats['mc_mean']:>10.3f} {stats['exp_mean']:>10.3f} {stats['filt_mean']:>10.3f} "
			f"{stats['mc_std']:>10.3f} {stats['exp_std']:>10.3f} {stats['filt_std']:>10.3f} "
			f"{stats['ks_exp_mc']:>8.3f} {stats['ks_filt_mc']:>8.3f}"
		)
