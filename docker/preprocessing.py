"""
preprocessing.py
----------------
Parses raw CSV input and engineers features for the quantile regression model.

Expected CSV columns (minimum):
    - net_load       : actual net load (MW)  [target]
    - DA_load        : day-ahead load forecast (MW)
    - DA_solar       : day-ahead solar forecast (MW)
    - DA_wind        : day-ahead wind forecast (MW)

The CSV must have a datetime index (UTC or naive → will be cast to UTC).
"""

import io
import numpy as np
import pandas as pd

# Canonical feature list used across train / predict
INPUT_FEATURES: list[str] = [
    "net_load_24",
    "net_load_25",
    "net_load_26",
    "DA_renewable",
    "DA_renewable_1",
    "DA_renewable_2",
    "DA_renewable_3",
    "DA_load",
    "DA_load_2",
    "hour_sin",
    "hour_cos",
]

REQUIRED_RAW_COLS: list[str] = ["net_load", "DA_load", "DA_solar", "DA_wind"]


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all engineered features to the dataframe in-place (returns copy)."""
    df = df.copy()

    # Lagged net load (Day-Ahead forecasting: minimum lag = 24 h)
    df["net_load_24"] = df["net_load"].shift(24)
    df["net_load_25"] = df["net_load"].shift(25)
    df["net_load_26"] = df["net_load"].shift(26)

    # Day-Ahead renewable aggregate and its lags
    df["DA_renewable"] = df["DA_solar"] + df["DA_wind"]
    df["DA_renewable_1"] = df["DA_renewable"].shift(1)
    df["DA_renewable_2"] = df["DA_renewable"].shift(2)
    df["DA_renewable_3"] = df["DA_renewable"].shift(3)

    # Lagged DA load
    df["DA_load_2"] = df["DA_load"].shift(2)

    # Cyclical hour encoding
    hours = df.index.hour
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24)

    return df


def parse_csv(contents: bytes) -> pd.DataFrame:
    """
    Parse raw CSV bytes into a feature-engineered DataFrame.

    Parameters
    ----------
    contents : bytes
        Raw bytes of the uploaded CSV file.

    Returns
    -------
    pd.DataFrame
        Hourly DataFrame with DatetimeIndex (UTC) and all engineered features.

    Raises
    ------
    ValueError
        If required columns are missing or the index cannot be parsed as dates.
    """
    df = pd.read_csv(io.BytesIO(contents), index_col=0, parse_dates=True)

    # Validate index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "The first column must be a parseable datetime index. "
            "Ensure the CSV has a datetime column as its first column."
        )

    # Normalise timezone to UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df.index.name = "datetime"

    # Validate required columns
    missing = [c for c in REQUIRED_RAW_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"The CSV must contain: {REQUIRED_RAW_COLS}"
        )

    # Resample to hourly frequency (fills gaps with NaN, which the model handles)
    df = df.asfreq("h")

    return _engineer_features(df)