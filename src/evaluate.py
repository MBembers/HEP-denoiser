from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from .utils import bootstrap_statistic


def ks_statistic(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
	"""Compute the two-sample KS statistic (max CDF difference)."""
	a = np.sort(sample_a)
	b = np.sort(sample_b)
	n = a.size
	m = b.size
	if n == 0 or m == 0:
		return float("nan")

	data_all = np.sort(np.concatenate([a, b]))
	cdf_a = np.searchsorted(a, data_all, side="right") / n
	cdf_b = np.searchsorted(b, data_all, side="right") / m
	return float(np.max(np.abs(cdf_a - cdf_b)))


@dataclass
class BenchmarkResult:
	per_feature: Dict[str, Dict[str, float]]
	score_summary: Dict[str, Dict[str, float]]
	retention_rate: float


def score_separation(scores_mc: np.ndarray, scores_exp: np.ndarray) -> Dict[str, float]:
	"""Quick separation summary between MC and EXP scores."""
	return {
		"mc_mean": float(np.mean(scores_mc)),
		"exp_mean": float(np.mean(scores_exp)),
		"mean_delta": float(np.mean(scores_mc) - np.mean(scores_exp)),
	}


def _feature_stats(values: np.ndarray) -> Dict[str, float]:
	return {
		"mean": float(np.mean(values)),
		"std": float(np.std(values)),
		"median": float(np.median(values)),
	}


def benchmark_datasets(
	df_mc: pd.DataFrame,
	df_exp: pd.DataFrame,
	df_filtered: pd.DataFrame,
	scores_mc: np.ndarray,
	scores_exp: np.ndarray,
	scores_filtered: np.ndarray,
	features: Iterable[str],
) -> BenchmarkResult:
	"""Benchmark distributions and denoiser scores."""
	per_feature: Dict[str, Dict[str, float]] = {}
	for feat in features:
		mc_vals = df_mc[feat].to_numpy(dtype=float)
		exp_vals = df_exp[feat].to_numpy(dtype=float)
		filt_vals = df_filtered[feat].to_numpy(dtype=float)

		per_feature[feat] = {
			"ks_exp_vs_mc": ks_statistic(exp_vals, mc_vals),
			"ks_filt_vs_mc": ks_statistic(filt_vals, mc_vals),
			"exp_mean": _feature_stats(exp_vals)["mean"],
			"mc_mean": _feature_stats(mc_vals)["mean"],
			"filt_mean": _feature_stats(filt_vals)["mean"],
		}

	score_summary = {
		"mc": bootstrap_statistic(scores_mc, np.mean, n_resamples=500),
		"exp": bootstrap_statistic(scores_exp, np.mean, n_resamples=500),
		"filtered": bootstrap_statistic(scores_filtered, np.mean, n_resamples=500),
	}

	retention_rate = float(len(df_filtered) / max(len(df_exp), 1))
	return BenchmarkResult(
		per_feature=per_feature,
		score_summary=score_summary,
		retention_rate=retention_rate,
	)


def sweep_retention_vs_ks(
	*,
	denoiser,
	df_mc: pd.DataFrame,
	df_exp: pd.DataFrame,
	probs_exp: np.ndarray,
	features: Iterable[str],
	quantiles: Iterable[float],
) -> pd.DataFrame:
	"""Sweep thresholds by EXP probability quantiles and compute average KS."""
	rows = []
	for q in quantiles:
		if not 0.0 < q < 1.0:
			continue
		threshold = float(np.quantile(probs_exp, q))
		df_filtered, _ = denoiser.filter_dataframe(df_exp, threshold)
		ks_values = []
		for feat in features:
			mc_vals = df_mc[feat].to_numpy(dtype=float)
			exp_vals = df_filtered[feat].to_numpy(dtype=float)
			ks_values.append(ks_statistic(exp_vals, mc_vals))
		avg_ks = float(np.nanmean(ks_values)) if ks_values else float("nan")
		retention = float(len(df_filtered) / max(len(df_exp), 1))
		rows.append({"quantile": q, "threshold": threshold, "retention": retention, "avg_ks": avg_ks})

	return pd.DataFrame(rows)
