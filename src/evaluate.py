from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve


@dataclass
class RocResult:
	fpr: np.ndarray
	tpr: np.ndarray
	thresholds: np.ndarray
	auc_score: float


def compute_roc(y_true: np.ndarray, y_score: np.ndarray) -> RocResult:
	fpr, tpr, thresholds = roc_curve(y_true, y_score)
	return RocResult(
		fpr=fpr,
		tpr=tpr,
		thresholds=thresholds,
		auc_score=float(auc(fpr, tpr)),
	)


def compute_significance_curve(
	*,
	y_true: np.ndarray,
	y_score: np.ndarray,
	n_sig: float,
	n_bkg: float,
) -> pd.DataFrame:
	"""Compute S/sqrt(S+B) as a function of score threshold."""
	roc = compute_roc(y_true, y_score)
	S = n_sig * roc.tpr
	B = n_bkg * roc.fpr
	metric = np.divide(S, np.sqrt(S + B), out=np.zeros_like(S), where=(S + B) > 0)
	return pd.DataFrame({"threshold": roc.thresholds, "significance": metric})
