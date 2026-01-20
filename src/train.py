from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .data_loader import ImpactParameterDataset
from .model import BDTDenoiser


@dataclass
class BDTTrainResult:
	denoiser: BDTDenoiser
	threshold: float
	prob_exp: pd.Series
	prob_mc: pd.Series
	prob_bkg: pd.Series
	metrics: dict[str, float]
	exp_df: pd.DataFrame
	mc_df: pd.DataFrame
	background_df: pd.DataFrame


def train_bdt_denoiser(
	*,
	features: Optional[list[str]] = None,
	background_window: tuple[float, float] = (5700.0, 5900.0),
	threshold: float = 0.95,
	model_output_dir: Path = Path("models"),
) -> BDTTrainResult:
	"""Train a BDT on MC vs background and save the model."""
	dataset = ImpactParameterDataset()
	dataset.load_experimental()
	dataset.load_mc()

	exp_df = dataset.to_dataframe("exp")
	mc_df = dataset.to_dataframe("mc")
	used_features = features or dataset.var_names

	low, high = background_window
	background_df = exp_df.query(f"~({low} < Xb_M < {high})").copy()

	denoiser = BDTDenoiser(features=tuple(used_features))
	denoiser.fit(mc_df, background_df)
	denoiser.save(model_output_dir)

	prob_exp = pd.Series(denoiser.predict_proba(exp_df), index=exp_df.index, name="BDT")
	prob_mc = pd.Series(denoiser.predict_proba(mc_df), index=mc_df.index, name="BDT")
	prob_bkg = pd.Series(denoiser.predict_proba(background_df), index=background_df.index, name="BDT")

	pred_mc = prob_mc >= threshold
	pred_bkg = prob_bkg >= threshold
	tp = float(pred_mc.sum())
	fn = float((~pred_mc).sum())
	fp = float(pred_bkg.sum())
	tn = float((~pred_bkg).sum())

	precision = tp / max(tp + fp, 1.0)
	recall = tp / max(tp + fn, 1.0)
	accuracy = (tp + tn) / max(tp + tn + fp + fn, 1.0)
	f1 = 2 * precision * recall / max(precision + recall, 1e-12)

	metrics = {
		"tp": tp,
		"fp": fp,
		"tn": tn,
		"fn": fn,
		"precision": precision,
		"recall": recall,
		"accuracy": accuracy,
		"f1": f1,
	}

	return BDTTrainResult(
		denoiser=denoiser,
		threshold=threshold,
		prob_exp=prob_exp,
		prob_mc=prob_mc,
		prob_bkg=prob_bkg,
		metrics=metrics,
		exp_df=exp_df,
		mc_df=mc_df,
		background_df=background_df,
	)
