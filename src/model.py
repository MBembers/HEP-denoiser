from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _as_numpy_matrix(
	df: pd.DataFrame,
	features: Iterable[str],
	*,
	return_mask: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
	subset = df.loc[:, list(features)].astype(float)
	subset = subset.replace([np.inf, -np.inf], np.nan)
	valid_mask = ~subset.isna().any(axis=1)
	data = subset.loc[valid_mask].to_numpy()
	if return_mask:
		return data, valid_mask.to_numpy()
	return data


def _shrink_covariance(cov: np.ndarray, shrinkage: float) -> np.ndarray:
	if shrinkage <= 0.0:
		return cov
	diag = np.diag(np.diag(cov))
	return (1.0 - shrinkage) * cov + shrinkage * diag


@dataclass
class GaussianDenoiser:
	"""Gaussian denoiser based on Mahalanobis distance to MC distribution."""

	features: Optional[Tuple[str, ...]] = None
	shrinkage: float = 1e-3

	mean_: Optional[np.ndarray] = None
	cov_: Optional[np.ndarray] = None
	inv_cov_: Optional[np.ndarray] = None

	def fit(self, df_mc: pd.DataFrame, features: Optional[Iterable[str]] = None) -> "GaussianDenoiser":
		used_features = tuple(features) if features is not None else self.features
		if not used_features:
			raise ValueError("features must be provided to fit the denoiser")

		data = _as_numpy_matrix(df_mc, used_features)
		if data.size == 0:
			raise ValueError("No valid rows remain after filtering NaNs/infs")

		mean = np.mean(data, axis=0)
		cov = np.cov(data, rowvar=False)
		cov = _shrink_covariance(cov, self.shrinkage)
		inv_cov = np.linalg.pinv(cov)

		self.features = used_features
		self.mean_ = mean
		self.cov_ = cov
		self.inv_cov_ = inv_cov
		return self

	def score(self, df: pd.DataFrame) -> np.ndarray:
		if self.mean_ is None or self.inv_cov_ is None or not self.features:
			raise ValueError("Model not fitted. Call fit() first.")
		data = _as_numpy_matrix(df, self.features)
		diff = data - self.mean_
		distances = np.einsum("ij,jk,ik->i", diff, self.inv_cov_, diff)
		return distances

	def threshold_by_quantile(self, scores: np.ndarray, quantile: float = 0.95) -> float:
		if not 0.0 < quantile < 1.0:
			raise ValueError("quantile must be between 0 and 1")
		return float(np.quantile(scores, quantile))

	def filter_dataframe(self, df: pd.DataFrame, threshold: float) -> Tuple[pd.DataFrame, np.ndarray]:
		if self.mean_ is None or self.inv_cov_ is None or not self.features:
			raise ValueError("Model not fitted. Call fit() first.")
		data, valid_mask = _as_numpy_matrix(df, self.features, return_mask=True)
		diff = data - self.mean_
		scores = np.einsum("ij,jk,ik->i", diff, self.inv_cov_, diff)
		keep_mask = scores <= threshold
		combined_mask = valid_mask.copy()
		combined_mask[valid_mask] = keep_mask
		filtered = df.loc[combined_mask].copy()
		return filtered, combined_mask


@dataclass
class MLPClassifierDenoiser:
	"""PyTorch MLP classifier to score MC-likeness of events."""

	features: Optional[Tuple[str, ...]] = None
	hidden_dim: int = 32
	learning_rate: float = 1e-3
	epochs: int = 50
	batch_size: int = 512
	random_state: int = 42
	device: Optional[str] = None

	mean_: Optional[np.ndarray] = None
	std_: Optional[np.ndarray] = None
	model_: Optional[nn.Module] = None

	def _standardize(self, x: np.ndarray) -> np.ndarray:
		if self.mean_ is None or self.std_ is None:
			raise ValueError("Model not fitted.")
		return (x - self.mean_) / self.std_

	def _init_model(self, input_dim: int) -> None:
		self.model_ = nn.Sequential(
			nn.Linear(input_dim, self.hidden_dim),
			nn.ReLU(),
			nn.Linear(self.hidden_dim, 1),
			nn.Sigmoid(),
		)

	def fit(
		self,
		df_mc: pd.DataFrame,
		df_exp: pd.DataFrame,
		features: Optional[Iterable[str]] = None,
		sample_size: Optional[int] = None,
	) -> "MLPClassifierDenoiser":
		used_features = tuple(features) if features is not None else self.features
		if not used_features:
			raise ValueError("features must be provided to fit the denoiser")

		mc = _as_numpy_matrix(df_mc, used_features)
		exp = _as_numpy_matrix(df_exp, used_features)

		if sample_size:
			rng = np.random.default_rng(self.random_state)
			if mc.shape[0] > sample_size:
				mc = mc[rng.choice(mc.shape[0], size=sample_size, replace=False)]
			if exp.shape[0] > sample_size:
				exp = exp[rng.choice(exp.shape[0], size=sample_size, replace=False)]

		x = np.vstack([mc, exp])
		y = np.concatenate([np.ones(mc.shape[0]), np.zeros(exp.shape[0])])

		mean = np.mean(x, axis=0)
		std = np.std(x, axis=0)
		std = np.where(std == 0, 1.0, std)

		self.mean_ = mean
		self.std_ = std
		self._init_model(x.shape[1])

		torch.manual_seed(self.random_state)
		device = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
		print(f"Training on device: {device}")
		self.model_.to(device)

		x_tensor = torch.from_numpy(self._standardize(x)).float()
		y_tensor = torch.from_numpy(y).float().unsqueeze(1)
		dataset = TensorDataset(x_tensor, y_tensor)
		loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

		criterion = nn.BCELoss()
		optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)

		for _ in range(self.epochs):
			self.model_.train()
			for xb, yb in loader:
				xb = xb.to(device)
				yb = yb.to(device)
				optimizer.zero_grad()
				preds = self.model_(xb)
				loss = criterion(preds, yb)
				loss.backward()
				optimizer.step()

		self.features = used_features
		return self

	def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
		if self.model_ is None:
			raise ValueError("Model not fitted. Call fit() first.")
		data = _as_numpy_matrix(df, self.features)
		x = torch.from_numpy(self._standardize(data)).float()
		device = next(self.model_.parameters()).device
		self.model_.eval()
		with torch.no_grad():
			probs = self.model_(x.to(device)).cpu().numpy().squeeze(axis=1)
		return probs

	def threshold_by_quantile(self, probs_mc: np.ndarray, retain_quantile: float = 0.95) -> float:
		if not 0.0 < retain_quantile < 1.0:
			raise ValueError("retain_quantile must be between 0 and 1")
		return float(np.quantile(probs_mc, 1.0 - retain_quantile))

	def filter_dataframe(self, df: pd.DataFrame, threshold: float) -> Tuple[pd.DataFrame, np.ndarray]:
		data, valid_mask = _as_numpy_matrix(df, self.features, return_mask=True)
		x = torch.from_numpy(self._standardize(data)).float()
		device = next(self.model_.parameters()).device
		self.model_.eval()
		with torch.no_grad():
			probs = self.model_(x.to(device)).cpu().numpy().squeeze(axis=1)
		keep_mask = probs >= threshold
		combined_mask = valid_mask.copy()
		combined_mask[valid_mask] = keep_mask
		filtered = df.loc[combined_mask].copy()
		return filtered, combined_mask
