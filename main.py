"""
main.py
-------
FastAPI application — Net Load Probabilistic Forecasting.

Single workflow endpoint:
  POST /forecast   Upload CSV → train → predict → return charts + predictions

Supporting endpoints:
  GET  /           Serve the web UI
  GET  /docs       Auto-generated OpenAPI docs (kept for dev convenience)
"""

import uuid
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .model import QuantileRegressionForecaster
from .plotting import (
    compute_pit,
    plot_cost_reliability,
    plot_forecast_intervals,
    plot_pit_histogram,
)
from .preprocessing import parse_csv

# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NetLoad Forecast API",
    description="Probabilistic net-load forecasting via quantile regression.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url=None,
)

STATIC_DIR = Path(__file__).parent / "static"
PLOTS_DIR = STATIC_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


# ---------------------------------------------------------------------------
def _resolve_cutoff(df: pd.DataFrame, train_end_date: Optional[str]) -> pd.Timestamp:
    if train_end_date:
        try:
            cutoff = pd.Timestamp(train_end_date, tz="UTC")
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Invalid train_end_date — use YYYY-MM-DD format.",
            )
    else:
        cutoff = df.index[int(len(df) * 0.8)]
    return cutoff


def _make_prediction_rows(pred_df: pd.DataFrame) -> list:
    return [
        {
            "timestamp": str(ts),
            "quantiles": {str(col): round(float(val), 2) for col, val in row.items()},
        }
        for ts, row in pred_df.iterrows()
    ]


# ---------------------------------------------------------------------------
@app.post("/forecast", tags=["Forecast"])
async def forecast(
    request: Request,
    file: UploadFile = File(...),
    train_end_date: Optional[str] = Form(None),
    use_backward_selection: bool = Form(True),
):
    """
    One-shot endpoint: upload CSV → train → predict → get charts.

    Returns JSON with URLs to 5 PNG charts and the full predictions array.
    """
    contents = await file.read()
    try:
        df = parse_csv(contents)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    cutoff = _resolve_cutoff(df, train_end_date)
    df_train = df[df.index <= cutoff]
    df_test  = df[df.index >  cutoff]

    if len(df_train) < 200:
        raise HTTPException(
            status_code=400,
            detail=f"Training set only has {len(df_train)} rows — need at least 200.",
        )
    if len(df_test) == 0:
        raise HTTPException(
            status_code=400,
            detail="No test rows found after the training cutoff. Adjust train_end_date.",
        )

    logger.info("Training on %d rows, testing on %d rows.", len(df_train), len(df_test))
    model = QuantileRegressionForecaster()
    try:
        model.fit(df_train, use_backward_selection=use_backward_selection)
    except Exception as exc:
        logger.exception("Training failed.")
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}")

    try:
        pred_test  = model.predict(df_test)
        pred_train = model.predict(df_train)
    except Exception as exc:
        logger.exception("Prediction failed.")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    # --- Charting ----------------------------------------------------------
    logger.info("Generating charts…")

    run_id = uuid.uuid4().hex
    output_plots_dir = PLOTS_DIR / run_id
    output_plots_dir.mkdir()

    plot_forecast_intervals(
        pred_test, df_test["net_load"],
        save_path=output_plots_dir / "forecast_test.png",
        title="Probabilistic Forecast — Test Period",
        max_points=len(pred_test),
    )
    plot_forecast_intervals(
        pred_train, df_train["net_load"],
        save_path=output_plots_dir / "forecast_train.png",
        title="Probabilistic Forecast — Training Period (first 500 h)",
        max_points=500,
    )

    pit_train_vals  = compute_pit(df_train["net_load"], pred_train)
    pit_test_vals   = compute_pit(df_test["net_load"],  pred_test)
    plot_pit_histogram(
        pit_train_vals,
        save_path=output_plots_dir / "pit_train.png",
        title="PIT Calibration — Training Set"
    )
    plot_pit_histogram(
        pit_test_vals,
        save_path=output_plots_dir / "pit_test.png",
        title="PIT Calibration — Test Set"
    )

    df_test_plot = df_test.copy()
    if "DA_renewable" not in df_test_plot.columns:
        df_test_plot["DA_renewable"] = df_test_plot["DA_solar"] + df_test_plot["DA_wind"]
    plot_cost_reliability(
        df_test_plot, pred_test, save_path=output_plots_dir / "cost_reliability.png"
    )

    def get_chart_url(name):
        return str(request.url_for('static', path=f"plots/{run_id}/{name}"))

    return {
        "train_period":    {"start": model.train_start, "end": model.train_end},
        "test_period":     {"start": str(df_test.index.min()), "end": str(df_test.index.max())},
        "n_train_samples": model.n_train_samples,
        "n_test_samples":  len(pred_test),
        "quantiles":       [round(q, 2) for q in model.quantiles],
        "charts": {
            "forecast_test":    get_chart_url("forecast_test.png"),
            "forecast_train":   get_chart_url("forecast_train.png"),
            "pit_train":        get_chart_url("pit_train.png"),
            "pit_test":         get_chart_url("pit_test.png"),
            "cost_reliability": get_chart_url("cost_reliability.png"),
        },
        "predictions": _make_prediction_rows(pred_test),
    }