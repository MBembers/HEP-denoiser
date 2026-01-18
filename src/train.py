from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .data_loader import ImpactParameterDataset
from .model import GaussianDenoiser, MLPClassifierDenoiser


@dataclass
class DenoiseResult:
	denoiser: GaussianDenoiser
	threshold: float
	scores_mc: np.ndarray
	scores_exp: np.ndarray
	scores_filtered: np.ndarray
	df_mc: pd.DataFrame
	df_exp: pd.DataFrame
	df_filtered: pd.DataFrame


@dataclass
class ClassifierDenoiseResult:
	denoiser: MLPClassifierDenoiser
	threshold: float
	probs_mc: np.ndarray
	probs_exp: np.ndarray
	probs_filtered: np.ndarray
	df_mc: pd.DataFrame
	df_exp: pd.DataFrame
	df_filtered: pd.DataFrame


def train_denoiser(
	*,
	quantile: float = 0.95,
	shrinkage: float = 1e-3,
	features: Optional[list[str]] = None,
) -> DenoiseResult:
	"""Fit the Gaussian denoiser on MC data and filter experimental data."""
	dataset = ImpactParameterDataset()
	dataset.load_experimental()
	dataset.load_mc()

	df_exp = dataset.to_dataframe("exp")
	df_mc = dataset.to_dataframe("mc")
	used_features = features or dataset.var_names

	denoiser = GaussianDenoiser(features=tuple(used_features), shrinkage=shrinkage)
	denoiser.fit(df_mc)

	scores_mc = denoiser.score(df_mc)
	scores_exp = denoiser.score(df_exp)
	threshold = denoiser.threshold_by_quantile(scores_mc, quantile=quantile)
	df_filtered, _ = denoiser.filter_dataframe(df_exp, threshold)
	scores_filtered = denoiser.score(df_filtered)

	return DenoiseResult(
		denoiser=denoiser,
		threshold=threshold,
		scores_mc=scores_mc,
		scores_exp=scores_exp,
		scores_filtered=scores_filtered,
		df_mc=df_mc,
		df_exp=df_exp,
		df_filtered=df_filtered,
	)


def train_classifier_denoiser(
	*,
	retain_quantile: float = 0.95,
	target_retention: Optional[float] = None,
	hidden_dim: int = 32,
	learning_rate: float = 1e-3,
	epochs: int = 50,
	batch_size: int = 512,
	sample_size: Optional[int] = None,
	features: Optional[list[str]] = None,
	device: Optional[str] = None,
) -> ClassifierDenoiseResult:
	"""Train an MLP classifier to separate MC vs EXP and filter MC-like EXP."""
	dataset = ImpactParameterDataset()
	dataset.load_experimental()
	dataset.load_mc()

	df_exp = dataset.to_dataframe("exp")
	df_mc = dataset.to_dataframe("mc")
	used_features = features or dataset.var_names

	denoiser = MLPClassifierDenoiser(
		features=tuple(used_features),
		hidden_dim=hidden_dim,
		learning_rate=learning_rate,
		epochs=epochs,
		batch_size=batch_size,
		device=device,
	)
	denoiser.fit(df_mc, df_exp, sample_size=sample_size)

	probs_mc = denoiser.predict_proba(df_mc)
	probs_exp = denoiser.predict_proba(df_exp)
	if target_retention is not None:
		if not 0.0 < target_retention < 1.0:
			raise ValueError("target_retention must be between 0 and 1")
		threshold = float(np.quantile(probs_exp, 1.0 - target_retention))
	else:
		threshold = denoiser.threshold_by_quantile(probs_mc, retain_quantile=retain_quantile)
	df_filtered, _ = denoiser.filter_dataframe(df_exp, threshold)
	probs_filtered = denoiser.predict_proba(df_filtered)

	return ClassifierDenoiseResult(
		denoiser=denoiser,
		threshold=threshold,
		probs_mc=probs_mc,
		probs_exp=probs_exp,
		probs_filtered=probs_filtered,
		df_mc=df_mc,
		df_exp=df_exp,
		df_filtered=df_filtered,
	)
