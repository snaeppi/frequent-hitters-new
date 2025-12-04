#!/usr/bin/env python3
"""Clean HTS data using Hit Dexter 3–style curation and split by assay format."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Set

import polars as pl
import typer
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

# Suppress RDKit warnings
RDLogger.logger().setLevel(RDLogger.CRITICAL)

app = typer.Typer(no_args_is_help=True, help="Clean HTS data and split by assay format.")

PROGRESS_COLUMNS = (
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("{task.completed}/{task.total}"),
    "[progress.percentage]{task.percentage:>3.0f}%",
    "•",
    TimeElapsedColumn(),
    "•",
    TimeRemainingColumn(),
)


# Elements allowed per Hit Dexter 3 methodology
ALLOWED_ATOMS: Set[int] = {
    1,  # H
    5,  # B
    6,  # C
    7,  # N
    8,  # O
    9,  # F
    14,  # Si
    15,  # P
    16,  # S
    17,  # Cl
    34,  # Se
    35,  # Br
    53,  # I
}

MIN_MW = 180.0
MAX_MW = 900.0


def _setup_logger(log_file: Path | None) -> logging.Logger:
    logger = logging.getLogger("assay_cleaning")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(levelname)s] %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


class ChemicalProcessor:
    def __init__(self) -> None:
        self.uncharger = rdMolStandardize.Uncharger()
        self.fragment_chooser = rdMolStandardize.LargestFragmentChooser()
        self.tautomer_enumerator = rdMolStandardize.TautomerEnumerator()

    def process_smiles(self, smi: Optional[str]) -> Optional[str]:
        """Apply neutralisation, desalting, tautomer canonicalisation, and filters."""
        if smi is None:
            return None

        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return None

            mol = self.uncharger.uncharge(mol)
            mol = self.fragment_chooser.choose(mol)
            mol = self.uncharger.uncharge(mol)
            mol = self.tautomer_enumerator.Canonicalize(mol)
            if mol is None:
                return None

            mw = Descriptors.ExactMolWt(mol)  # type: ignore[attr-defined]
            if not (MIN_MW <= mw <= MAX_MW):
                return None

            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() not in ALLOWED_ATOMS:
                    return None

            canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
            try:
                if Chem.MolFromSmiles(canonical) is None:
                    return None
            except Exception:
                return None
            return canonical
        except Exception:
            return None


def read_table(path: Path) -> pl.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        return pl.read_csv(path)
    if ext == ".parquet":
        return pl.read_parquet(path)
    raise ValueError(f"Unsupported input extension '{ext}'. Use .csv or .parquet")


def write_table(df: pl.DataFrame, path: Path) -> None:
    ext = path.suffix.lower()
    if ext == ".csv":
        df.write_csv(path)
    elif ext == ".parquet":
        df.write_parquet(path)
    else:
        raise ValueError(f"Unsupported output extension '{ext}'. Use .csv or .parquet")


def rename_cols(df: pl.DataFrame, smiles_col: str, assay_col: str, active_col: str) -> pl.DataFrame:
    mapping = {
        smiles_col: "smiles",
        assay_col: "assay_id",
        active_col: "active",
    }
    safe_mapping = {
        src: tgt for src, tgt in mapping.items() if src in df.columns and tgt not in df.columns
    }
    return df.rename(safe_mapping)


@app.command("clean")
def clean_split(
    hts_file: Path = typer.Option(..., "--hts-file", help="Path to HTS data file (.csv or .parquet)."),
    assay_props_file: Path = typer.Option(..., "--assay-props-file", help="Path to assay properties file (.csv or .parquet)."),
    id_to_smiles_file: Path | None = typer.Option(
        None,
        "--id-to-smiles-file",
        help="Optional path to ID-to-SMILES mapping file.",
    ),
    id_col: str = typer.Option("compound_id", "--id-col", help="Column name for compound ID."),
    smiles_col: str = typer.Option("smiles", "--smiles-col", help="Column name for SMILES."),
    assay_col: str = typer.Option("assay_id", "--assay-col", help="Column name for assay ID."),
    active_col: str = typer.Option("active", "--active-col", help="Column name for activity flag."),
    assay_format_col: str = typer.Option("assay_format", "--assay-format-col", help="Column name for assay format."),
    biochemical_format: str = typer.Option("biochemical", "--biochemical-format", help="Label for biochemical assays."),
    cellular_format: str = typer.Option("cellular", "--cellular-format", help="Label for cellular assays."),
    score_col: str | None = typer.Option(
        None,
        "--score-col",
        help="Continuous score column to derive binary 'active' labels when no explicit activity is present.",
    ),
    score_threshold: float = typer.Option(
        3.0,
        "--score-threshold",
        help="Absolute threshold applied to --score-col to derive 'active'.",
    ),
    biochemical_out: Path = typer.Option(..., "--biochemical-out", help="Output file for biochemical subset."),
    cellular_out: Path = typer.Option(..., "--cellular-out", help="Output file for cellular subset."),
    rename_columns: bool = typer.Option(
        False,
        "--rename-cols",
        help="Rename columns to canonical names (smiles, assay_id, active).",
    ),
    log_file: Path | None = typer.Option(None, "--log-file", help="Optional log file path."),
) -> None:
    """Clean HTS data and split into biochemical and cellular subsets."""
    logger = _setup_logger(log_file)

    hts_df = read_table(hts_file)
    assay_props_df = read_table(assay_props_file)
    logger.info("Loaded HTS data: %s rows", f"{hts_df.height:,}")
    logger.info("Loaded assay properties: %s rows", f"{assay_props_df.height:,}")

    if id_to_smiles_file:
        id_to_smiles_df = read_table(id_to_smiles_file)
        logger.info("Loaded ID-to-SMILES mapping: %s rows", f"{id_to_smiles_df.height:,}")
    else:
        id_to_smiles_df = hts_df.select([id_col, smiles_col]).unique()
        logger.info("Built ID-to-SMILES mapping from HTS data")

    original_smiles_count = id_to_smiles_df.height
    processor = ChemicalProcessor()

    canonical_map: dict[str, Optional[str]] = {}
    smiles_list = id_to_smiles_df[smiles_col].to_list()
    with Progress(*PROGRESS_COLUMNS) as progress:
        task_id = progress.add_task("Processing structures", total=len(smiles_list))
        for smi in smiles_list:
            canonical_map[smi] = processor.process_smiles(smi)
            progress.advance(task_id)
        progress.update(task_id, description="[green]Structures processed")

    id_to_smiles_df = id_to_smiles_df.with_columns(
        pl.col(smiles_col)
        .map_elements(lambda x: canonical_map.get(x), return_dtype=pl.Utf8)
        .alias("canonical_smiles")
    )

    id_to_smiles_df = id_to_smiles_df.filter(pl.col("canonical_smiles").is_not_null())
    retained_smiles_count = id_to_smiles_df.height

    dropped_count = original_smiles_count - retained_smiles_count
    logger.info("Original unique compounds: %s", f"{original_smiles_count:,}")
    logger.info("Dropped compounds (filters/errors): %s", f"{dropped_count:,}")
    logger.info("Retained valid compounds: %s", f"{retained_smiles_count:,}")

    clean_hts_df = hts_df.join(
        id_to_smiles_df.select([id_col, "canonical_smiles"]),
        on=id_col,
        how="inner",
    )

    if "smiles" not in clean_hts_df.columns:
        clean_hts_df = clean_hts_df.rename({"canonical_smiles": "smiles"})
    else:
        clean_hts_df = clean_hts_df.drop("smiles").rename({"canonical_smiles": "smiles"})

    logger.info("HTS rows remaining after structure cleaning: %s", f"{clean_hts_df.height:,}")

    clean_hts_df = clean_hts_df.join(
        assay_props_df.select([assay_col, assay_format_col]),
        on=assay_col,
        how="inner",
    )

    if rename_columns:
        clean_hts_df = rename_cols(
            clean_hts_df,
            smiles_col="smiles",
            assay_col=assay_col,
            active_col=active_col,
        )
        logger.info("Columns renamed to canonical names")

    if "active" not in clean_hts_df.columns:
        if score_col is not None and score_col in clean_hts_df.columns:
            logger.info(
                "Deriving 'active' column from R-score column '%s' using threshold %s",
                score_col,
                score_threshold,
            )
            clean_hts_df = clean_hts_df.with_columns(
                (pl.col(score_col).abs() >= score_threshold).cast(pl.Int8).alias("active")
            )
        else:
            raise typer.BadParameter(
                "No 'active' column present after cleaning and no valid --score-col configured. "
                "Provide an activity column via --active-col/--rename-cols or specify --score-col."
            )
    else:
        clean_hts_df = clean_hts_df.with_columns(pl.col("active").cast(pl.Int8))

    biochemical_df = clean_hts_df.filter(pl.col(assay_format_col) == biochemical_format)
    cellular_df = clean_hts_df.filter(pl.col(assay_format_col) == cellular_format)

    logger.info("Biochemical subset: %s rows", f"{biochemical_df.height:,}")
    logger.info("Cellular subset: %s rows", f"{cellular_df.height:,}")

    write_table(biochemical_df, biochemical_out)
    write_table(cellular_df, cellular_out)
    logger.info("Wrote biochemical subset to %s", biochemical_out)
    logger.info("Wrote cellular subset to %s", cellular_out)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
