"""Murcko scaffold splitting utilities."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable
from pathlib import Path

import polars as pl
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from . import viz


def _normalize_scaffold_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Return a two-column DataFrame of smiles → scaffold assignments."""
    if df.is_empty():
        return pl.DataFrame(
            {
                "smiles": pl.Series([], dtype=pl.Utf8),
                "scaffold_smiles": pl.Series([], dtype=pl.Utf8),
            }
        )

    columns = df.columns
    if "smiles" not in columns:
        raise ValueError("Scaffold table must include a 'smiles' column.")

    if "scaffold_smiles" in columns:
        scaffold_col = "scaffold_smiles"
    elif "scaffold" in columns:
        scaffold_col = "scaffold"
    elif len(columns) == 2:
        scaffold_col = next(col for col in columns if col != "smiles")
    else:
        raise ValueError(
            "Scaffold table must include either a 'scaffold' column or exactly two columns (smiles + scaffold)."
        )

    normalized = (
        df.select(
            [
                pl.col("smiles").cast(pl.Utf8, strict=False).alias("smiles"),
                pl.when(pl.col(scaffold_col).is_null())
                .then(pl.lit(None, dtype=pl.Utf8))
                .otherwise(pl.col(scaffold_col).cast(pl.Utf8, strict=False))
                .alias("scaffold_smiles"),
            ]
        )
        .unique(subset=["smiles"], keep="first")
        .sort("smiles")
    )
    return normalized


class ScaffoldStore:
    """Cache of Bemis-Murcko scaffolds keyed by SMILES strings."""

    def __init__(
        self,
        mapping: dict[str, str | None] | None = None,
    ) -> None:
        self._mapping: dict[str, str | None] = dict(mapping or {})
        self._dirty = False

    @classmethod
    def from_frame(cls, df: pl.DataFrame) -> ScaffoldStore:
        normalized = _normalize_scaffold_frame(df)
        mapping = {row[0]: row[1] for row in normalized.iter_rows()}
        return cls(mapping)

    @classmethod
    def from_file(cls, path: Path) -> ScaffoldStore:
        if path.suffix.lower() in {".parquet", ".pq"}:
            frame = pl.read_parquet(path)
        else:
            frame = pl.read_csv(path)
        return cls.from_frame(frame)

    def ensure(self, smiles: Iterable[str]) -> list[str | None]:
        smiles_list = list(smiles)
        missing = {smi for smi in smiles_list if smi not in self._mapping}
        if missing:
            newly_computed = {smi: _murcko_scaffold(smi) for smi in missing}
            self._mapping.update(newly_computed)
            self._dirty = True
        return [self._mapping[smi] for smi in smiles_list]

    def update_from_frame(self, df: pl.DataFrame) -> None:
        normalized = _normalize_scaffold_frame(df)
        updates = {row[0]: row[1] for row in normalized.iter_rows()}
        if updates:
            self._mapping.update(updates)
            self._dirty = True

    def is_empty(self) -> bool:
        return not self._mapping

    def to_frame(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "smiles": list(self._mapping.keys()),
                "scaffold_smiles": list(self._mapping.values()),
            }
        ).sort("smiles")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = self.to_frame()
        if path.suffix.lower() in {".parquet", ".pq"}:
            frame.write_parquet(path)
        else:
            frame.write_csv(path)
        self._dirty = False

    @property
    def dirty(self) -> bool:
        return self._dirty


def _murcko_scaffold(smiles: str) -> str | None:
    """Return the Bemis-Murcko scaffold or None if unavailable."""
    if smiles is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return scaffold or None


def _compute_targets(
    fractions: dict[str, float],
    total_count: int,
) -> dict[str, int]:
    """Convert fractional split definitions to integer targets."""
    raw = {split: float(total_count) * float(frac) for split, frac in fractions.items()}
    base = {split: int(math.floor(val)) for split, val in raw.items()}
    remainder = total_count - sum(base.values())
    if remainder > 0:
        residuals = sorted(
            ((val - base[split], split) for split, val in raw.items()),
            reverse=True,
        )
        for _, split in residuals[:remainder]:
            base[split] += 1
    # Ensure no negative targets (can happen if fractions omit splits)
    for split in base:
        base[split] = max(base[split], 0)
    return base


def _summarize_split_counts(df: pl.DataFrame, *, split_column: str) -> list[dict[str, object]]:
    """Return per-split counts and unique scaffolds for a given split column."""
    if split_column not in df.columns:
        raise KeyError(f"Split column '{split_column}' not found.")
    summary_df = (
        df.group_by(split_column)
        .agg(
            [
                pl.len().alias("compound_count"),
                pl.col("scaffold_smiles").n_unique().alias("unique_scaffolds"),
            ]
        )
        .sort(split_column)
    )
    return summary_df.to_dicts()


def _attach_scaffolds(
    compound_metadata: pl.DataFrame,
    scaffold_store: ScaffoldStore,
) -> pl.DataFrame:
    """Ensure a scaffold column exists using the shared scaffold store."""
    scaffolds = scaffold_store.ensure(compound_metadata["smiles"].to_list())
    scaffold_series = pl.Series("scaffold_smiles", scaffolds, dtype=pl.Utf8)
    return compound_metadata.with_columns(scaffold_series)


def _ensure_regression_flag(df: pl.DataFrame) -> pl.DataFrame:
    """Attach a regression_eligible boolean column."""
    if "regression_eligible" in df.columns:
        return df.with_columns(pl.col("regression_eligible").cast(pl.Boolean))
    if "passes_reliability_filter" in df.columns:
        return df.with_columns(
            pl.col("passes_reliability_filter").cast(pl.Boolean).alias("regression_eligible")
        )
    raise ValueError(
        "compound_metadata must include either 'regression_eligible' "
        "or 'passes_reliability_filter'."
    )


def _assign_scaffold_split_column(
    compound_df: pl.DataFrame,
    *,
    fractions: dict[str, float],
    seed: int,
    split_column: str,
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Assign deterministic Murcko-scaffold splits to the provided subset."""
    if compound_df.is_empty():
        raise ValueError("Cannot assign splits to an empty compound set.")

    total_fraction = sum(fractions.values())
    if not math.isclose(total_fraction, 1.0, rel_tol=1e-6):
        raise ValueError(f"Split fractions must sum to 1.0 (got {total_fraction:.6f}).")

    splits = list(fractions.keys())
    total_compounds = compound_df.height

    scaffold_groups: dict[str, list[str]] = {}
    group_keys: dict[str, str] = {}
    for smi, scaffold in zip(
        compound_df["smiles"].to_list(),
        compound_df["scaffold_smiles"].to_list(),
        strict=False,
    ):
        key = scaffold if scaffold is not None else f"__NOSCAF__::{smi}"
        scaffold_groups.setdefault(key, []).append(smi)
        group_keys[smi] = key

    targets = _compute_targets(fractions, total_compounds)
    assigned_counts = {split: 0 for split in splits}

    rng = random.Random(seed)  # noqa: S311 - deterministic split shuffling only
    scaffold_items = list(scaffold_groups.items())
    rng.shuffle(scaffold_items)

    split_assignments: dict[str, str] = {}
    for key, smiles_list in scaffold_items:
        best_split = max(
            splits,
            key=lambda split: (
                targets.get(split, 0) - assigned_counts[split],
                -assigned_counts[split],
            ),
        )
        split_assignments[key] = best_split
        assigned_counts[best_split] += len(smiles_list)

    split_values = [split_assignments[group_keys[smi]] for smi in compound_df["smiles"]]
    split_series = pl.Series(split_column, split_values, dtype=pl.Utf8)
    split_df = compound_df.with_columns(split_series)

    summary = {
        "seed": seed,
        "fractions": fractions,
        "total_compounds": total_compounds,
        "split_counts": _summarize_split_counts(split_df, split_column=split_column),
        "unique_scaffolds_total": int(split_df["scaffold_smiles"].n_unique()),
    }
    return split_df, summary


def assign_multiseed_splits(
    compound_metadata: pl.DataFrame,
    output_dir: Path,
    *,
    fractions: dict[str, float],
    seeds: list[int],
    enable_plots: bool,
    scaffold_store: ScaffoldStore | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, object]]:
    """
    Assign scaffold-aware splits for regression and multi-task models across multiple seeds.

    Returns two DataFrames (regression-only and multi-task) that both contain per-seed
    `split_seed<seed>` columns (train/val/test). Test membership is identical per seed across the two
    frames.
    """
    if not seeds:
        raise ValueError("At least one seed must be provided for splitting.")

    output_dir.mkdir(parents=True, exist_ok=True)
    if scaffold_store is None:
        scaffold_store = ScaffoldStore()

    required = {"train", "val", "test"}
    missing = required.difference(fractions)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(
            f"Split fractions must include train, val, and test (missing: {missing_str})."
        )

    train_frac = float(fractions["train"])
    val_frac = float(fractions["val"])
    reg_fractions = {"train": train_frac, "val": val_frac, "test": float(fractions["test"])}

    train_val_total = train_frac + val_frac
    if train_val_total <= 0:
        raise ValueError("Train and val fractions must be positive.")
    multitask_fractions = {
        "train": train_frac / train_val_total,
        "val": val_frac / train_val_total,
    }

    base_df = _ensure_regression_flag(compound_metadata)
    base_df = _attach_scaffolds(base_df, scaffold_store)
    base_df = base_df.with_columns(pl.col("regression_eligible").fill_null(False).cast(pl.Boolean))

    summary: dict[str, object] = {
        "fractions": reg_fractions,
        "multitask_fractions": multitask_fractions,
        "seeds": {},
    }

    reg_mappings: dict[int, dict[str, str]] = {}
    mt_mappings: dict[int, dict[str, str]] = {}
    shared_tests: dict[int, set[str]] = {}

    for seed in seeds:
        reg_col = f"split_seed{seed}_reg"
        mt_col = f"split_seed{seed}_mt"

        regression_pool = base_df.filter(pl.col("regression_eligible"))
        regression_split_df, _reg_stats = _assign_scaffold_split_column(
            regression_pool,
            fractions=reg_fractions,
            seed=seed,
            split_column=reg_col,
        )
        reg_mapping: dict[str, str] = dict(
            zip(
                regression_split_df["smiles"].to_list(),
                regression_split_df[reg_col].to_list(),
                strict=False,
            )
        )
        reg_mappings[seed] = reg_mapping

        test_smiles = {
            smi
            for smi, split in zip(
                regression_split_df["smiles"].to_list(),
                regression_split_df[reg_col].to_list(),
                strict=False,
            )
            if split == "test"
        }
        shared_tests[seed] = test_smiles

        multitask_pool = base_df.filter(~pl.col("smiles").is_in(test_smiles))
        if multitask_pool.is_empty():
            raise ValueError(
                "All compounds were assigned to the shared test set; no data left for train/val."
            )

        multitask_split_df, _ = _assign_scaffold_split_column(
            multitask_pool,
            fractions=multitask_fractions,
            seed=seed,
            split_column=mt_col,
        )
        mt_mapping = dict(
            zip(
                multitask_split_df["smiles"].to_list(),
                multitask_split_df[mt_col].to_list(),
                strict=False,
            )
        )
        mt_mappings[seed] = mt_mapping

        seed_summary = {
            "seed": seed,
            "regression_split_counts": _summarize_split_counts(
                regression_split_df.rename({reg_col: f"split_seed{seed}"}),
                split_column=f"split_seed{seed}",
            ),
            "multitask_split_counts": _summarize_split_counts(
                multitask_split_df.rename({mt_col: f"split_seed{seed}"}),
                split_column=f"split_seed{seed}",
            ),
            "shared_test_compounds": len(test_smiles),
        }
        summary["seeds"][str(seed)] = seed_summary

    reg_split_df = base_df.filter(pl.col("regression_eligible")).with_columns(
        [
            pl.col("smiles")
            .replace(reg_mappings[seed], default=None)
            .alias(f"split_seed{seed}")
            for seed in seeds
        ]
    )

    mt_split_df = base_df.with_columns(
        [
            pl.when(pl.col("smiles").is_in(shared_tests[seed]))
            .then(pl.lit("test"))
            .otherwise(pl.col("smiles").replace(mt_mappings[seed], default=None))
            .alias(f"split_seed{seed}")
            for seed in seeds
        ]
    )

    with (output_dir / "split_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    if enable_plots and seeds:
        plot_dir = output_dir / "plots"
        plot_dir.mkdir(exist_ok=True)
        stacked = pl.concat(
            [
                reg_split_df.select(
                    [
                        pl.lit(str(seed)).alias("seed"),
                        pl.col(f"split_seed{seed}").alias("split"),
                        pl.col("screens"),
                    ]
                )
                for seed in seeds
            ],
            how="vertical",
        )
        viz.plot_compound_screens_distribution(
            stacked,
            plot_dir / "compound_screens_distribution.png",
            plot_dir / "compound_screens_distribution.csv",
            seed_column="seed",
        )

    return reg_split_df, mt_split_df, summary
