"""
model.py
--------
Quantile Regression Forecaster for net load.

Key design:
  - 19 quantiles linearly spaced between 1/20 and 19/20
    (≈ 0.05, 0.10, … 0.95)
  - Optional backward feature elimination per quantile (p-value threshold 5%)
  - joblib serialisation → persisted under MODELS_DIR (/models by default)
"""

import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from .preprocessing import INPUT_FEATURES

logger = logging.getLogger(__name__)

MODELS_DIR = Path("/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

N_QUANTILES = 19


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def backward_elimination_quantreg(
    data: pd.DataFrame,
    target: str,
    features: List[str],
    q: float,
    max_iter: int = 500,
    threshold: float = 0.05,
) -> List[str]:
    """
    Iteratively remove the least significant feature (highest p-value > threshold)
    until all remaining features are significant.

    Parameters
    ----------
    data       : training DataFrame
    target     : name of the response column
    features   : candidate feature list
    q          : quantile level
    max_iter   : max iterations for the IRLS solver
    threshold  : p-value significance threshold

    Returns
    -------
    List of selected feature names (may be empty if nothing is significant).
    """
    remaining = list(features)

    while remaining:
        formula = f"{target} ~ {' + '.join(remaining)} - 1"
        try:
            model = smf.quantreg(formula, data=data).fit(
                q=q, max_iter=max_iter, disp=False
            )
        except Exception as exc:
            logger.warning("quantreg fit failed during backward elimination: %s", exc)
            break

        p_values = model.pvalues
        max_p = p_values.max()

        if max_p > threshold:
            worst = p_values.idxmax()
            logger.debug("q=%.2f  removing '%s' (p=%.4f)", q, worst, max_p)
            remaining.remove(worst)
        else:
            break  # all features are significant

    return remaining


# ---------------------------------------------------------------------------
# Forecaster
# ---------------------------------------------------------------------------

class QuantileRegressionForecaster:
    """
    Trains one quantile-regression model per quantile level and exposes
    `predict` to return a DataFrame of quantile forecasts.
    """

    def __init__(self, n_quantiles: int = N_QUANTILES):
        self.n_quantiles = n_quantiles
        self.quantiles: np.ndarray = np.linspace(
            1 / (n_quantiles + 1), n_quantiles / (n_quantiles + 1), n_quantiles
        )
        self.models: Dict[float, object] = {}
        self.selected_features: Dict[float, List[str]] = {}
        self.is_fitted: bool = False
        self.train_start: Optional[str] = None
        self.train_end: Optional[str] = None
        self.n_train_samples: int = 0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        df_train: pd.DataFrame,
        use_backward_selection: bool = True,
    ) -> "QuantileRegressionForecaster":
        """
        Fit one quantile-regression model per quantile level.

        Parameters
        ----------
        df_train               : feature-engineered training DataFrame
        use_backward_selection : if True, run backward p-value elimination
                                 per quantile before the final fit

        Returns
        -------
        self
        """
        df_clean = df_train.dropna(subset=["net_load"] + INPUT_FEATURES)
        if len(df_clean) == 0:
            raise ValueError("No complete rows in training data after dropping NaN.")

        self.train_start = str(df_clean.index.min())
        self.train_end = str(df_clean.index.max())
        self.n_train_samples = len(df_clean)

        for q in self.quantiles:
            q_round = round(q, 2)
            logger.info("Training quantile q=%.2f …", q_round)

            if use_backward_selection:
                features = backward_elimination_quantreg(
                    df_clean, "net_load", INPUT_FEATURES, q
                )
            else:
                features = list(INPUT_FEATURES)

            # Fallback: if backward selection removed everything, use all features
            if not features:
                logger.warning(
                    "Backward selection removed all features at q=%.2f; "
                    "falling back to full feature set.",
                    q_round,
                )
                features = list(INPUT_FEATURES)

            self.selected_features[q_round] = features

            formula = f"net_load ~ {' + '.join(features)} - 1"
            self.models[q_round] = smf.quantreg(formula, data=df_clean).fit(
                q=q, max_iter=25_000, disp=False
            )

        self.is_fitted = True
        logger.info("Training complete. %d models fitted.", len(self.models))
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate quantile predictions for the given DataFrame.

        Rows with NaN in any feature column are silently dropped.

        Returns
        -------
        pd.DataFrame  shape (n_valid_rows, n_quantiles)
                      columns are quantile levels (float), index is datetime.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict().")

        df_clean = df.dropna(subset=INPUT_FEATURES)
        if len(df_clean) == 0:
            raise ValueError("No valid rows to predict after dropping NaN in features.")

        raw_predictions: Dict[float, pd.Series] = {}
        for q_round, model in self.models.items():
            raw_predictions[q_round] = model.predict(df_clean)

        pred_df = pd.DataFrame(raw_predictions, index=df_clean.index)

        # Enforce monotonicity by sorting quantile values row-wise
        sorted_values = np.sort(pred_df.values, axis=1)
        pred_df = pd.DataFrame(
            sorted_values,
            columns=sorted(raw_predictions.keys()),
            index=df_clean.index,
        )
        return pred_df

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, model_id: str) -> Path:
        """Serialise the forecaster to disk and return the file path."""
        path = MODELS_DIR / f"{model_id}.joblib"
        joblib.dump(self, path)
        logger.info("Model saved → %s", path)
        return path

    @classmethod
    def load(cls, model_id: str) -> "QuantileRegressionForecaster":
        """Deserialise a previously saved forecaster."""
        path = MODELS_DIR / f"{model_id}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"No model found with id='{model_id}'.")
        return joblib.load(path)

    @staticmethod
    def list_saved() -> List[str]:
        """Return the list of saved model IDs."""
        return [p.stem for p in sorted(MODELS_DIR.glob("*.joblib"))]