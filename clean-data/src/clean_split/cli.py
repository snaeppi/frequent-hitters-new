#!/usr/bin/env python3
"""
Clean HTS data using Hit Dexter 3–style curation and split by assay format.

This module:
- Processes SMILES (neutralize, desalt, canonicalize tautomers, filter by MW/elements).
- Builds an ID-to-SMILES mapping on the fly.
- Joins clean SMILES back to HTS data.
- Splits cleaned data into biochemical and cellular subsets.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Set

import polars as pl
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rich.progress import track

# Suppress RDKit warnings
RDLogger.logger().setLevel(RDLogger.CRITICAL)

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

            mw = Descriptors.ExactMolWt(mol)
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean HTS data using Hit Dexter 3 protocols and split by format."
    )
    parser.add_argument(
        "--hts-file",
        required=True,
        type=Path,
        help="Path to HTS data file (.csv or .parquet)",
    )
    parser.add_argument(
        "--assay-props-file",
        required=True,
        type=Path,
        help="Path to assay properties file (.csv or .parquet)",
    )
    parser.add_argument(
        "--id-to-smiles-file",
        type=Path,
        default=None,
        help="Optional path to ID-to-SMILES mapping file",
    )
    parser.add_argument("--id-col", default="compound_id", help="Column name for compound ID")
    parser.add_argument("--smiles-col", default="smiles", help="Column name for SMILES")
    parser.add_argument("--assay-col", default="assay_id", help="Column name for assay ID")
    parser.add_argument("--active-col", default="active", help="Column name for activity flag")
    parser.add_argument(
        "--assay-format-col",
        default="assay_format",
        help="Column name for assay format",
    )
    parser.add_argument(
        "--biochemical-format",
        default="biochemical",
        help="Label for biochemical assays",
    )
    parser.add_argument(
        "--cellular-format",
        default="cellular",
        help="Label for cellular assays",
    )
    parser.add_argument(
        "--score-col",
        default=None,
        help=(
            "Optional column containing a continuous score used to derive the "
            "binary 'active' label when no explicit activity column is present."
        ),
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=3.0,
        help=(
            "Threshold applied to --score-col; values strictly greater than this "
            "threshold are treated as active (1)."
        ),
    )
    parser.add_argument(
        "--biochemical-out",
        required=True,
        type=Path,
        help="Output file for biochemical subset",
    )
    parser.add_argument(
        "--cellular-out",
        required=True,
        type=Path,
        help="Output file for cellular subset",
    )
    parser.add_argument(
        "--rename-cols",
        action="store_true",
        help="Enable renaming columns to canonical names",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional log file path",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logger = logging.getLogger("clean_split")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(levelname)s] %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if args.log_file:
        fh = logging.FileHandler(args.log_file, mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    hts_df = read_table(args.hts_file)
    assay_props_df = read_table(args.assay_props_file)
    logger.info("Loaded HTS data: %s rows", f"{hts_df.height:,}")
    logger.info("Loaded assay properties: %s rows", f"{assay_props_df.height:,}")

    if args.id_to_smiles_file:
        id_to_smiles_df = read_table(args.id_to_smiles_file)
        logger.info("Loaded ID-to-SMILES mapping: %s rows", f"{id_to_smiles_df.height:,}")
    else:
        id_to_smiles_df = hts_df.select([args.id_col, args.smiles_col]).unique()
        logger.info("Built ID-to-SMILES mapping from HTS data")

    original_smiles_count = id_to_smiles_df.height
    processor = ChemicalProcessor()

    smiles_list = id_to_smiles_df[args.smiles_col].to_list()
    canonical_map: dict[str, Optional[str]] = {}
    for smi in track(smiles_list, description="Processing structures"):
        canonical_map[smi] = processor.process_smiles(smi)

    id_to_smiles_df = id_to_smiles_df.with_columns(
        pl.col(args.smiles_col)
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
        id_to_smiles_df.select([args.id_col, "canonical_smiles"]),
        on=args.id_col,
        how="inner",
    )

    if "smiles" not in clean_hts_df.columns:
        clean_hts_df = clean_hts_df.rename({"canonical_smiles": "smiles"})
    else:
        clean_hts_df = clean_hts_df.drop("smiles").rename({"canonical_smiles": "smiles"})

    logger.info(
        "HTS rows remaining after structure cleaning: %s",
        f"{clean_hts_df.height:,}",
    )

    clean_hts_df = clean_hts_df.join(
        assay_props_df.select([args.assay_col, args.assay_format_col]),
        on=args.assay_col,
        how="inner",
    )

    if args.rename_cols:
        clean_hts_df = rename_cols(
            clean_hts_df,
            smiles_col="smiles",
            assay_col=args.assay_col,
            active_col=args.active_col,
        )
        logger.info("Columns renamed to canonical names")

    if "active" not in clean_hts_df.columns:
        if args.score_col is not None and args.score_col in clean_hts_df.columns:
            logger.info(
                "Deriving 'active' column from score column '%s' using threshold %s",
                args.score_col,
                args.score_threshold,
            )
            clean_hts_df = clean_hts_df.with_columns(
                (pl.col(args.score_col) > args.score_threshold)
                .cast(pl.Int8)
                .alias("active")
            )
        else:
            raise ValueError(
                "No 'active' column present after cleaning and no valid --score-col configured. "
                "Provide an activity column via --active-col/--rename-cols or specify --score-col."
            )
    else:
        clean_hts_df = clean_hts_df.with_columns(pl.col("active").cast(pl.Int8))

    biochemical_df = clean_hts_df.filter(
        pl.col(args.assay_format_col) == args.biochemical_format
    )
    cellular_df = clean_hts_df.filter(
        pl.col(args.assay_format_col) == args.cellular_format
    )

    logger.info("Biochemical subset: %s rows", f"{biochemical_df.height:,}")
    logger.info("Cellular subset: %s rows", f"{cellular_df.height:,}")

    write_table(biochemical_df, args.biochemical_out)
    write_table(cellular_df, args.cellular_out)
    logger.info("Wrote biochemical subset to %s", args.biochemical_out)
    logger.info("Wrote cellular subset to %s", args.cellular_out)


if __name__ == "__main__":
    main()

