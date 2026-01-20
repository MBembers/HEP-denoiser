from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


def _as_numpy_matrix(df: pd.DataFrame, features: Iterable[str]) -> np.ndarray:
	data = df.loc[:, list(features)].astype(float)
	data = data.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
	return data.to_numpy()


@dataclass
class BDTDenoiser:
	"""BDT denoiser using XGBoost on full feature set."""

	features: Optional[Tuple[str, ...]] = None
	model: Optional[XGBClassifier] = None

	def fit(
		self,
		df_signal: pd.DataFrame,
		df_background: pd.DataFrame,
		features: Optional[Iterable[str]] = None,
		n_estimators: int = 200,
		max_depth: int = 4,
		learning_rate: float = 0.05,
		subsample: float = 0.9,
		colsample_bytree: float = 0.9,
		random_state: int = 42,
	) -> "BDTDenoiser":
		used_features = tuple(features) if features is not None else self.features
		if not used_features:
			raise ValueError("features must be provided to fit the denoiser")

		x_sig = _as_numpy_matrix(df_signal, used_features)
		x_bkg = _as_numpy_matrix(df_background, used_features)
		if x_sig.size == 0 or x_bkg.size == 0:
			raise ValueError("Empty signal or background after cleaning NaNs/infs")

		y_sig = np.ones(x_sig.shape[0], dtype=int)
		y_bkg = np.zeros(x_bkg.shape[0], dtype=int)
		x = np.vstack([x_sig, x_bkg])
		y = np.concatenate([y_sig, y_bkg])

		model = XGBClassifier(
			n_estimators=n_estimators,
			max_depth=max_depth,
			learning_rate=learning_rate,
			subsample=subsample,
			colsample_bytree=colsample_bytree,
			eval_metric="logloss",
			random_state=random_state,
		)
		model.fit(x, y)

		self.features = used_features
		self.model = model
		return self

	def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
		if self.model is None or not self.features:
			raise ValueError("Model not fitted. Call fit() first.")
		x = _as_numpy_matrix(df, self.features)
		return self.model.predict_proba(x)[:, 1]

	def filter_dataframe(self, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
		probs = self.predict_proba(df)
		mask = probs >= threshold
		filtered = df.iloc[: len(mask)].loc[mask].copy()
		return filtered

	def save(self, output_dir: Path, name: str = "bdt_model") -> Path:
		if self.model is None:
			raise ValueError("Model not fitted. Call fit() first.")
		output_dir.mkdir(parents=True, exist_ok=True)
		model_path = output_dir / f"{name}.json"
		self.model.get_booster().save_model(str(model_path))
		features_path = output_dir / f"{name}_features.txt"
		features_path.write_text("\n".join(self.features or []))
		return model_path
