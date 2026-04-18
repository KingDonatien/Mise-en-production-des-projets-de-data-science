"""
schemas.py
----------
Pydantic models for all FastAPI request and response bodies.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# /train
# ---------------------------------------------------------------------------

class TrainResponse(BaseModel):
    model_id: str = Field(..., description="Unique identifier for the trained model")
    n_train_samples: int = Field(..., description="Number of complete rows used for training")
    train_start: str = Field(..., description="First timestamp in training data")
    train_end: str = Field(..., description="Last timestamp in training data")
    quantiles: List[float] = Field(..., description="Quantile levels fitted (e.g. 0.05, 0.10, …)")
    features_per_quantile: Dict[str, List[str]] = Field(
        ..., description="Features selected by backward elimination for each quantile"
    )


# ---------------------------------------------------------------------------
# /predict  &  /forecast
# ---------------------------------------------------------------------------

class PredictionRow(BaseModel):
    timestamp: str = Field(..., description="ISO-8601 datetime of the prediction")
    quantiles: Dict[str, float] = Field(
        ..., description="Predicted value for each quantile level"
    )


class PredictResponse(BaseModel):
    model_id: str
    n_predictions: int
    predictions: List[PredictionRow]


class ForecastResponse(BaseModel):
    model_id: str = Field(..., description="Auto-generated model ID for the trained model")
    train_period: Dict[str, str] = Field(..., description="{'start': ..., 'end': ...}")
    test_period: Dict[str, str] = Field(..., description="{'start': ..., 'end': ...}")
    n_train_samples: int
    n_test_samples: int
    predictions: List[PredictionRow]


# ---------------------------------------------------------------------------
# /models
# ---------------------------------------------------------------------------

class ModelsListResponse(BaseModel):
    models: List[str] = Field(..., description="List of stored model IDs")


# ---------------------------------------------------------------------------
# Generic error
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    detail: str