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
        return pl.DataFrame({"smiles": pl.Series([], dtype=pl.Utf8), "scaffold_smiles": pl.Series([], dtype=pl.Utf8)})

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
        mapping = {
            row[0]: row[1]
            for row in normalized.iter_rows()
        }
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
        updates = {
            row[0]: row[1]
            for row in normalized.iter_rows()
        }
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


def assign_scaffold_splits(
    compound_metadata: pl.DataFrame,
    output_dir: Path,
    *,
    fractions: dict[str, float],
    seed: int,
    enable_plots: bool,
    scaffold_store: ScaffoldStore | None = None,
) -> tuple[pl.DataFrame, dict[str, object]]:
    """
    Assign deterministic Murcko-scaffold splits and persist summary artifacts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    smiles_series = compound_metadata["smiles"]
    if scaffold_store is None:
        scaffold_store = ScaffoldStore()

    scaffolds = scaffold_store.ensure(smiles_series.to_list())
    scaffold_series = pl.Series("scaffold_smiles", scaffolds, dtype=pl.Utf8)

    compound_with_scaffold = compound_metadata.with_columns(scaffold_series)
    total_compounds = compound_with_scaffold.height

    total_fraction = sum(fractions.values())
    if not math.isclose(total_fraction, 1.0, rel_tol=1e-6):
        raise ValueError(f"Split fractions must sum to 1.0 (got {total_fraction:.6f}).")

    splits = list(fractions.keys())

    scaffold_groups: dict[str, list[str]] = {}
    group_keys: dict[str, str] = {}
    for smi, scaffold in zip(
        compound_with_scaffold["smiles"].to_list(),
        compound_with_scaffold["scaffold_smiles"].to_list(), strict=False,
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

    split_column = [
        split_assignments[group_keys[smi]] for smi in compound_with_scaffold["smiles"]
    ]
    split_series = pl.Series("split", split_column, dtype=pl.Utf8)
    split_map_df = compound_with_scaffold.with_columns(split_series)

    split_counts_df = (
        split_map_df.group_by("split")
        .agg(
            [
                pl.len().alias("compound_count"),
                pl.col("scaffold_smiles").n_unique().alias("unique_scaffolds"),
            ]
        )
        .sort("split")
    )

    split_counts = split_counts_df.to_dicts()

    summary = {
        "seed": seed,
        "fractions": fractions,
        "total_compounds": total_compounds,
        "split_counts": split_counts,
        "unique_scaffolds_total": int(split_map_df["scaffold_smiles"].n_unique()),
    }

    with (output_dir / "split_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    if enable_plots:
        plot_dir = output_dir / "plots"
        plot_dir.mkdir(exist_ok=True)
        viz.plot_compound_screens_distribution(
            split_map_df.select(["split", "screens"]),
            plot_dir / "compound_screens_distribution.png",
            plot_dir / "compound_screens_distribution.csv",
        )

    return split_map_df, summary
