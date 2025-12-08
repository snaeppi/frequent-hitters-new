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
    excluded_hit_rate_mask = assay_df["excluded_due_to_hit_rate"].to_numpy()
    excluded_min_screens_mask = assay_df["excluded_due_to_min_screens"].to_numpy()
    retained_mask = ~(excluded_hit_rate_mask | excluded_min_screens_mask)

    retained_values = values[retained_mask]
    excluded_hit_rate_values = values[excluded_hit_rate_mask]
    excluded_min_screens_values = values[excluded_min_screens_mask]

    if values.size == 0:
        raise ValueError("Assay DataFrame empty; cannot plot hit rate distribution.")

    vmin, vmax = float(values.min()), float(values.max())
    if vmin == vmax:
        vmax = vmin + 1e-3
    bin_edges = np.linspace(vmin, vmax, bins + 1)
    all_counts, _ = np.histogram(values, bins=bin_edges)
    retained_counts, _ = np.histogram(retained_values, bins=bin_edges)
    excluded_hit_rate_counts, _ = np.histogram(excluded_hit_rate_values, bins=bin_edges)
    excluded_min_screens_counts, _ = np.histogram(excluded_min_screens_values, bins=bin_edges)
    excluded_counts = excluded_hit_rate_counts + excluded_min_screens_counts

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        values,
        bins=bin_edges,
        color="#4f81bd",
        alpha=0.35,
        edgecolor="black",
        label="All assays",
    )
    if excluded_min_screens_values.size:
        ax.hist(
            excluded_min_screens_values,
            bins=bin_edges,
            color="#f2a341",
            alpha=0.85,
            edgecolor="#a66200",
            label="Excluded: min screens",
        )
    if excluded_hit_rate_values.size:
        ax.hist(
            excluded_hit_rate_values,
            bins=bin_edges,
            color="#c0504d",
            alpha=0.8,
            edgecolor="darkred",
            label="Excluded: hit-rate outlier",
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
            "count_excluded_hit_rate": excluded_hit_rate_counts,
            "count_excluded_min_screens": excluded_min_screens_counts,
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
    seed_column: str | None = None,
) -> None:
    """Histogram of screens per compound stratified by split."""
    _ensure_parent(out_png)
    if split_map_df.is_empty():
        raise ValueError("Split map empty; cannot plot compound screens distribution.")

    df = split_map_df
    df = df.filter(pl.col("split").is_not_null())
    if seed_column is not None and seed_column in df.columns:
        df = df.filter(pl.col(seed_column).is_not_null())

    if df.is_empty():
        raise ValueError("Split map empty after filtering null split values.")

    splits = sorted(df["split"].unique().to_list())
    seeds = (
        sorted(df[seed_column].unique().to_list()) if seed_column is not None and seed_column in df.columns else []
    )
    color_map = plt.colormaps["tab10"].resampled(max(len(splits), 1))

    all_screens = df["screens"].to_numpy()
    bin_edges = np.histogram_bin_edges(all_screens, bins=bins)

    fig, ax = plt.subplots(figsize=(10, 6))
    csv_rows = []

    for idx, split_value in enumerate(splits):
        split_df = df.filter(pl.col("split") == split_value)
        if split_df.is_empty():
            continue
        color = color_map(0.0 if len(splits) == 1 else idx / (len(splits) - 1))

        if seeds:
            per_seed_counts = []
            for seed in seeds:
                seed_subset = split_df.filter(pl.col(seed_column) == seed)
                if seed_subset.is_empty():
                    continue
                screens = seed_subset["screens"].to_numpy()
                counts, _ = np.histogram(screens, bins=bin_edges)
                per_seed_counts.append(counts)
                csv_rows.append(
                    pl.DataFrame(
                        {
                            "split": split_value,
                            seed_column: str(seed),
                            "bin_left": bin_edges[:-1],
                            "bin_right": bin_edges[1:],
                            "count": counts,
                        }
                    )
                )

            if not per_seed_counts:
                continue

            stacked = np.vstack(per_seed_counts)
            min_counts = stacked.min(axis=0)
            max_counts = stacked.max(axis=0)
            mean_counts = stacked.mean(axis=0)

            ax.step(
                bin_edges[:-1],
                mean_counts,
                where="post",
                color=color,
                label=split_value,
            )
            ax.fill_between(
                bin_edges[:-1],
                min_counts,
                max_counts,
                step="post",
                color=color,
                alpha=0.15,
            )

            csv_rows.append(
                pl.DataFrame(
                    {
                        "split": split_value,
                        seed_column: "aggregate",
                        "bin_left": bin_edges[:-1],
                        "bin_right": bin_edges[1:],
                        "count_min": min_counts,
                        "count_max": max_counts,
                        "count_mean": mean_counts,
                    }
                )
            )
        else:
            screens = split_df["screens"].to_numpy()
            counts, _ = np.histogram(screens, bins=bin_edges)
            ax.step(
                bin_edges[:-1],
                counts,
                where="post",
                color=color,
                label=split_value,
            )
            csv_rows.append(
                pl.DataFrame(
                    {
                        "split": split_value,
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
        stacked = pl.concat(csv_rows, how="diagonal_relaxed")
        _write_csv(stacked, out_csv)
    else:
        empty_payload = {
            "split": [],
            "bin_left": [],
            "bin_right": [],
            "count": [],
        }
        if seed_column is not None:
            empty_payload[seed_column] = []
            empty_payload.update({"count_min": [], "count_max": [], "count_mean": []})
        empty_df = pl.DataFrame(empty_payload)
        _write_csv(empty_df, out_csv)
