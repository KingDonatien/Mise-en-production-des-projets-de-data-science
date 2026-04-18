"""
plotting.py
-----------
Replicates every visualisation from the research notebooks as base64-encoded
PNG strings suitable for embedding directly in HTML <img> tags.

Charts produced:
  1. forecast_intervals   – nested prediction intervals + actual (train & test)
  2. pit_histogram        – PIT (probability integral transform) calibration check
  3. cost_reliability     – reserve cost vs under-coverage trade-off (test only)
"""

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless – no display needed inside Docker
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Shared style ──────────────────────────────────────────────────────────────
PALETTE = {
    "bg":        "#0d1117",
    "panel":     "#161b22",
    "border":    "#30363d",
    "text":      "#e6edf3",
    "subtext":   "#8b949e",
    "accent":    "#58a6ff",
    "orange":    "#f0883e",
    "red":       "#ff7b72",
    "green":     "#3fb950",
    "grid":      "#21262d",
}

def _apply_dark_style(fig: plt.Figure, axes) -> None:
    """Apply a consistent dark theme to a figure and its axes."""
    fig.patch.set_facecolor(PALETTE["bg"])
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    for ax in axes:
        ax.set_facecolor(PALETTE["panel"])
        ax.tick_params(colors=PALETTE["subtext"], labelsize=9)
        ax.xaxis.label.set_color(PALETTE["subtext"])
        ax.yaxis.label.set_color(PALETTE["subtext"])
        ax.title.set_color(PALETTE["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["border"])
        ax.grid(True, color=PALETTE["grid"], linewidth=0.6, alpha=0.8)

def _save_fig(fig: plt.Figure, path: Path) -> None:
    """Save a matplotlib figure to a file."""
    fig.savefig(path, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ── Chart 1: Forecast Intervals ───────────────────────────────────────────────

def plot_forecast_intervals(
    pred_df: pd.DataFrame,
    actual: pd.Series,
    save_path: Path,
    title: str = "Net Load – Probabilistic Forecast",
    max_points: int = 500,
) -> None:
    """
    Nested prediction-interval fan chart matching the notebook style.

    Parameters
    ----------
    pred_df   : DataFrame of shape (n, 19) with quantile columns (float)
    actual    : Series of observed net_load values, same index as pred_df
    save_path : path to save the PNG image
    title     : chart title
    max_points: cap display to the first N rows for readability
    """
    pred_plot = pred_df.iloc[:max_points]
    actual_plot = actual.reindex(pred_plot.index)
    cols = sorted(pred_plot.columns)   # quantile floats, ascending
    vals = pred_plot[cols].values       # shape (T, 19)

    fig, ax = plt.subplots(figsize=(14, 5))
    _apply_dark_style(fig, ax)

    # Nested intervals (index pairs within 19-quantile array: 0↔18, 1↔17, 2↔16, 3↔15)
    interval_defs = [
        (0, 18, 0.10, "90 %"),
        (1, 17, 0.18, "80 %"),
        (2, 16, 0.28, "70 %"),
        (3, 15, 0.40, "60 %"),
    ]
    for lo, hi, alpha, _ in interval_defs:
        ax.fill_between(
            pred_plot.index,
            vals[:, lo],
            vals[:, hi],
            color=PALETTE["orange"],
            alpha=alpha,
            linewidth=0,
        )

    # Median (index 9 → Q0.50)
    ax.plot(
        pred_plot.index, vals[:, 9],
        color=PALETTE["orange"], linewidth=1.8, label="Median (Q0.50)", zorder=3,
    )

    # Actual
    if actual_plot.notna().any():
        ax.plot(
            actual_plot.index, actual_plot.values,
            color=PALETTE["accent"], linewidth=1.4,
            alpha=0.9, label="Actual net load", zorder=4,
        )

    # Legend
    orange_patch = mpatches.Patch(color=PALETTE["orange"], alpha=0.5,
                                   label="Prediction intervals (60–90 %)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles + [orange_patch],
        labels + ["Prediction intervals (60–90 %)"],
        facecolor=PALETTE["panel"], edgecolor=PALETTE["border"],
        labelcolor=PALETTE["text"], fontsize=9, loc="upper right",
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("Net Load (MW)", fontsize=10)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    _save_fig(fig, save_path)


# ── Chart 2: PIT Histogram ────────────────────────────────────────────────────

def plot_pit_histogram(cdf_values: np.ndarray, save_path: Path, title: str = "PIT Histogram") -> None:
    """
    Probability Integral Transform calibration histogram.
    A perfectly calibrated model produces a Uniform(0,1) distribution.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    _apply_dark_style(fig, ax)

    n_bins = 20
    ax.hist(
        cdf_values, bins=n_bins, density=True,
        color=PALETTE["accent"], alpha=0.65, edgecolor=PALETTE["border"],
        linewidth=0.6, label="PIT distribution",
    )
    ax.axhline(1.0, color=PALETTE["red"], linestyle="--", linewidth=1.8,
               label="Perfect calibration (Uniform)")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 2.2)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Cumulative probability  F_t(y)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.legend(facecolor=PALETTE["panel"], edgecolor=PALETTE["border"],
              labelcolor=PALETTE["text"], fontsize=9)
    fig.tight_layout()
    _save_fig(fig, save_path)


# ── Chart 3: Cost / Reliability trade-off ─────────────────────────────────────

def plot_cost_reliability(
    df_test: pd.DataFrame,
    pred_df: pd.DataFrame,
    save_path: Path,
    avg_reserve_price: float = 1.0,   # € / MW / h  (set to 1 if unknown → shows MW directly)
) -> None:
    """
    Dual-axis chart:  average reserve level (left)  vs  under-coverage frequency (right)
    for ε = 0.01 … 0.15, matching the notebook figure.

    Parameters
    ----------
    df_test          : test DataFrame containing 'net_load', 'DA_load', 'DA_renewable'
    pred_df          : quantile prediction DataFrame (columns = quantile floats)
    save_path        : path to save the PNG image
    avg_reserve_price: average market price of reserves (€/MW/h); use 1 for unit-less
    """
    epsilons = [round(i / 100, 2) for i in range(1, 16)]
    cols = sorted(pred_df.columns)  # ascending quantiles

    forecast_error = df_test["net_load"] - (
        df_test["DA_load"] - df_test.get("DA_renewable", pd.Series(0, index=df_test.index))
    )
    # Align indices
    forecast_error = forecast_error.reindex(pred_df.index)

    avg_reserves = []
    under_cov = []

    for eps in epsilons:
        target_q = round(1 - eps, 2)
        # Find the closest quantile column
        closest_col = min(cols, key=lambda c: abs(c - target_q))
        R = pred_df[closest_col] - (
            df_test["DA_load"].reindex(pred_df.index)
            - df_test.get("DA_renewable", pd.Series(0, index=df_test.index)).reindex(pred_df.index)
        )
        avg_reserves.append(R.mean())
        under_cov.append((forecast_error > R).mean())

    cost = [r * avg_reserve_price for r in avg_reserves]

    fig, ax1 = plt.subplots(figsize=(9, 4))
    _apply_dark_style(fig, ax1)

    ax1.plot(epsilons, cost, color=PALETTE["accent"], linewidth=2, marker="o",
             markersize=4, label="Avg reserve (MW or k€/h)")
    ax1.set_xlabel("Target risk level  ε", fontsize=10)
    ylabel1 = "Avg reserve cost (€/h)" if avg_reserve_price != 1.0 else "Avg reserve level (MW)"
    ax1.set_ylabel(ylabel1, color=PALETTE["accent"], fontsize=10)
    ax1.tick_params(axis="y", labelcolor=PALETTE["accent"])

    ax2 = ax1.twinx()
    ax2.set_facecolor(PALETTE["panel"])
    ax2.plot(epsilons, under_cov, color=PALETTE["orange"], linewidth=2,
             linestyle="--", marker="s", markersize=4, label="Under-coverage freq.")
    ax2.set_ylabel("Under-coverage frequency", color=PALETTE["orange"], fontsize=10)
    ax2.tick_params(axis="y", labelcolor=PALETTE["orange"])
    for spine in ax2.spines.values():
        spine.set_edgecolor(PALETTE["border"])

    ax1.set_title("Cost–Reliability Trade-off for Reserve Dimensioning",
                  fontsize=12, fontweight="bold", pad=10)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               facecolor=PALETTE["panel"], edgecolor=PALETTE["border"],
               labelcolor=PALETTE["text"], fontsize=9)

    fig.tight_layout()
    _save_fig(fig, save_path)


# ── Helper: compute PIT values ────────────────────────────────────────────────

def compute_pit(actual: pd.Series, pred_df: pd.DataFrame) -> np.ndarray:
    """
    Compute PIT (CDF) values for a set of quantile predictions.

    Parameters
    ----------
    actual  : observed net_load (aligned with pred_df)
    pred_df : quantile predictions, columns = sorted quantile floats

    Returns
    -------
    np.ndarray of PIT values ∈ [0, 1]
    """
    n = pred_df.shape[1]
    sorted_vals = np.sort(pred_df.values, axis=1)
    y = actual.reindex(pred_df.index).values
    pit = np.array([
        np.searchsorted(sorted_vals[i], y[i]) / n
        for i in range(len(y))
    ])
    return pit