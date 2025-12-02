"""Visualization utilities with paired CSV exports."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def _ensure_parent(path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_csv(df: pl.DataFrame, path: Path | None) -> None:
    if path is None:
        return
    _ensure_parent(path)
    df.write_csv(path)


def plot_assay_hit_rate_distribution(
    assay_df: pl.DataFrame,
    mean: float,
    std: float,
    k: float,
    out_png: Path | None,
    out_csv: Path | None,
    *,
    bins: int = 30,
) -> None:
    """Histogram of assay hit rates with exclusion thresholds."""
    _ensure_parent(out_png)
    values = assay_df["hit_rate"].to_numpy()
    excluded_mask = assay_df["excluded_flag"].to_numpy()
    retained_values = values[~excluded_mask]
    excluded_values = values[excluded_mask]

    if values.size == 0:
        raise ValueError("Assay DataFrame empty; cannot plot hit rate distribution.")

    vmin, vmax = float(values.min()), float(values.max())
    if vmin == vmax:
        vmax = vmin + 1e-3
    bin_edges = np.linspace(vmin, vmax, bins + 1)
    all_counts, _ = np.histogram(values, bins=bin_edges)
    retained_counts, _ = np.histogram(retained_values, bins=bin_edges)
    excluded_counts, _ = np.histogram(excluded_values, bins=bin_edges)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        values,
        bins=bin_edges,
        color="#4f81bd",
        alpha=0.7,
        edgecolor="black",
        label="All assays",
    )
    if excluded_values.size:
        ax.hist(
            excluded_values,
            bins=bin_edges,
            color="#c0504d",
            alpha=0.8,
            edgecolor="darkred",
            label="Excluded assays",
        )

    ax.axvline(mean, color="green", linestyle="--", linewidth=2, label=f"Mean = {mean:.3f}")
    if std > 0:
        threshold = mean + k * std
        ax.axvline(
            threshold,
            color="red",
            linestyle=":",
            linewidth=2,
            label=rf"$\mu + {k:.2f}\sigma$ = {threshold:.3f}",
        )
    else:
        threshold = None

    ax.set_xlabel("Hit rate")
    ax.set_ylabel("Number of assays")
    ax.legend(loc="best")
    ax.set_title("Assay hit rate distribution")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    if out_png is not None:
        fig.savefig(out_png, dpi=300)
    plt.close(fig)

    csv_df = pl.DataFrame(
        {
            "bin_left": bin_edges[:-1],
            "bin_right": bin_edges[1:],
            "count_all": all_counts,
            "count_retained": retained_counts,
            "count_excluded": excluded_counts,
        }
    )
    csv_df = csv_df.with_columns(
        [
            pl.lit(mean).alias("mean"),
            pl.lit(std).alias("std"),
            pl.lit(k).alias("std_k"),
            pl.lit(threshold).alias("threshold"),
        ]
    )
    _write_csv(csv_df, out_csv)


def plot_assay_coverage_hist(
    retained_assays: pl.DataFrame,
    out_png: Path | None,
    out_csv: Path | None,
    *,
    bins: int = 30,
) -> None:
    """Histogram of screens per retained assay."""
    _ensure_parent(out_png)
    screens = retained_assays["n_screens"].to_numpy()
    if screens.size == 0:
        raise ValueError("No retained assays for coverage histogram.")
    bin_edges = np.linspace(float(screens.min()), float(screens.max()), bins + 1)
    counts, _ = np.histogram(screens, bins=bin_edges)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        screens,
        bins=bin_edges,
        color="#4f81bd",
        edgecolor="black",
        alpha=0.8,
    )
    ax.set_xlabel("Screens per assay")
    ax.set_ylabel("Number of assays")
    ax.set_title("Retained assay coverage")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    if out_png is not None:
        fig.savefig(out_png, dpi=300)
    plt.close(fig)

    csv_df = pl.DataFrame(
        {
            "bin_left": bin_edges[:-1],
            "bin_right": bin_edges[1:],
            "count": counts,
        }
    )
    _write_csv(csv_df, out_csv)


def plot_assay_positive_rate_hist(
    retained_assays: pl.DataFrame,
    out_png: Path | None,
    out_csv: Path | None,
    *,
    bins: int = 30,
) -> None:
    """Histogram of hit rates for retained assays only."""
    _ensure_parent(out_png)
    hit_rates = retained_assays["hit_rate"].to_numpy()
    if hit_rates.size == 0:
        raise ValueError("No retained assays for positive rate histogram.")
    vmin, vmax = float(hit_rates.min()), float(hit_rates.max())
    if vmin == vmax:
        vmax = vmin + 1e-3
    bin_edges = np.linspace(vmin, vmax, bins + 1)
    counts, _ = np.histogram(hit_rates, bins=bin_edges)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        hit_rates,
        bins=bin_edges,
        color="#4f81bd",
        edgecolor="black",
        alpha=0.8,
    )
    ax.set_xlabel("Hit rate")
    ax.set_ylabel("Number of assays")
    ax.set_title("Retained assay hit rate distribution")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    if out_png is not None:
        fig.savefig(out_png, dpi=300)
    plt.close(fig)

    csv_df = pl.DataFrame(
        {
            "bin_left": bin_edges[:-1],
            "bin_right": bin_edges[1:],
            "count": counts,
        }
    )
    _write_csv(csv_df, out_csv)


def plot_retention_curve(
    retention_df: pl.DataFrame,
    chosen_threshold: float | None,
    out_png: Path | None,
    out_csv: Path | None,
    *,
    threshold_column: str = "min_screens",
    x_label: str = "Minimum screens per compound",
    chosen_label: str | None = None,
) -> None:
    """Plot retention fraction as a function of the supplied threshold column."""
    _ensure_parent(out_png)
    if retention_df.is_empty():
        raise ValueError("Retention DataFrame empty; cannot plot curve.")

    threshold_values = retention_df[threshold_column].to_numpy()
    fractions = retention_df["fraction_retained"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        threshold_values,
        fractions,
        marker="o",
        color="#4f81bd",
    )
    if chosen_threshold is not None:
        label_text = chosen_label or f"Chosen = {chosen_threshold}"
        ax.axvline(chosen_threshold, color="red", linestyle="--", label=label_text)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Fraction retained")
    ax.set_ylim(0.0, 1.05)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best")
    ax.grid(alpha=0.2)
    ax.set_title("Retention curve")
    plt.tight_layout()
    if out_png is not None:
        fig.savefig(out_png, dpi=300)
    plt.close(fig)

    _write_csv(retention_df, out_csv)


def plot_compound_screens_distribution(
    split_map_df: pl.DataFrame,
    out_png: Path | None,
    out_csv: Path | None,
    *,
    bins: int = 25,
) -> None:
    """Histogram of screens per compound stratified by split."""
    _ensure_parent(out_png)
    if split_map_df.is_empty():
        raise ValueError("Split map empty; cannot plot compound screens distribution.")

    splits = split_map_df["split"].unique().to_list()
    color_map = plt.colormaps["tab10"].resampled(max(len(splits), 1))

    fig, ax = plt.subplots(figsize=(10, 6))
    csv_rows = []
    for idx, split in enumerate(splits):
        subset = split_map_df.filter(pl.col("split") == split)
        screens = subset["screens"].to_numpy()
        if screens.size == 0:
            continue
        bin_edges = np.histogram_bin_edges(screens, bins=bins)
        counts, bin_edges = np.histogram(screens, bins=bin_edges)
        color = color_map(0.0 if len(splits) == 1 else idx / (len(splits) - 1))
        ax.step(
            bin_edges[:-1],
            counts,
            where="post",
            color=color,
            label=split,
        )
        csv_rows.append(
            pl.DataFrame(
                {
                    "split": split,
                    "bin_left": bin_edges[:-1],
                    "bin_right": bin_edges[1:],
                    "count": counts,
                }
            )
        )

    ax.set_xlabel("Screens selected")
    ax.set_ylabel("Compound count")
    ax.set_title("Screens per compound by split")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    if out_png is not None:
        fig.savefig(out_png, dpi=300)
    plt.close(fig)

    if csv_rows:
        stacked = pl.concat(csv_rows)
        _write_csv(stacked, out_csv)
    else:
        empty_df = pl.DataFrame(
            {"split": [], "bin_left": [], "bin_right": [], "count": []}
        )
        _write_csv(empty_df, out_csv)
