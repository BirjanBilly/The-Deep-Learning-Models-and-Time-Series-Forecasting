from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_training_history(history: pd.DataFrame, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    if "validation_tail_relative_error" in history:
        axis.plot(history["epoch"], history["validation_tail_relative_error"], label="Validation tail relative error")
    if "generator_loss" in history:
        axis.plot(history["epoch"], history["generator_loss"], label="Generator objective")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Metric")
    axis.set_title("Tail-sensitive scenario training")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_rank_frequency(rank: pd.DataFrame, path: str | Path, max_strategies: int = 6) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    strategies = rank["strategy"].drop_duplicates().tolist()[:max_strategies]
    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    for strategy in strategies:
        part = rank[rank["strategy"] == strategy]
        axis.plot(part["quantile"], part["real_pnl_quantile"], label=f"Real: {strategy}")
        axis.plot(part["quantile"], part["generated_pnl_quantile"], linestyle="--", label=f"Generated: {strategy}")
    axis.set_xscale("log")
    axis.set_xlabel("Lower-tail quantile level")
    axis.set_ylabel("Strategy PnL quantile")
    axis.set_title("Rank-frequency comparison")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_matrix_pair(real: np.ndarray, generated: np.ndarray, path: str | Path, title: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.1))
    vmin = float(min(np.nanmin(real), np.nanmin(generated)))
    vmax = float(max(np.nanmax(real), np.nanmax(generated)))
    images = []
    images.append(axes[0].imshow(real, vmin=vmin, vmax=vmax, aspect="auto"))
    axes[0].set_title("Observed")
    images.append(axes[1].imshow(generated, vmin=vmin, vmax=vmax, aspect="auto"))
    axes[1].set_title("Generated")
    for axis in axes:
        axis.set_xlabel("Column")
        axis.set_ylabel("Row")
    figure.suptitle(title)
    figure.colorbar(images[-1], ax=axes, shrink=0.82)
    figure.subplots_adjust(left=0.08, right=0.92, bottom=0.12, top=0.84, wspace=0.28)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_forecast_fan(predictions: pd.DataFrame, path: str | Path, horizon: int = 1) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = predictions[predictions["horizon"] == horizon].copy()
    if frame.empty:
        return
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date")
    quantile_columns = sorted(
        [column for column in frame.columns if column.startswith("q_")],
        key=lambda value: float(value[2:]),
    )
    q_values = np.asarray([float(column[2:]) for column in quantile_columns])
    median = quantile_columns[int(np.argmin(np.abs(q_values - 0.5)))]
    figure, axis = plt.subplots(figsize=(10.0, 4.6))
    axis.plot(frame["date"], frame["target"], label="Realized return")
    axis.plot(frame["date"], frame[median], label="Scenario median")
    for lower_probability, upper_probability in ((0.05, 0.95), (0.025, 0.975)):
        lower = quantile_columns[int(np.argmin(np.abs(q_values - lower_probability)))]
        upper = quantile_columns[int(np.argmin(np.abs(q_values - upper_probability)))]
        axis.fill_between(frame["date"], frame[lower], frame[upper], alpha=0.18)
    axis.axhline(0.0, linewidth=0.8)
    axis.set_title(f"Tail-GAN scenario forecast fan: {horizon}-day return")
    axis.set_xlabel("Forecast origin")
    axis.set_ylabel("Cumulative log return")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
