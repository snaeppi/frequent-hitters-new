#!/usr/bin/env python3
"""Clean HTS data using Hit Dexter 3-style curation and split by assay format."""

from __future__ import annotations

import json
import logging
import multiprocessing
from pathlib import Path
from typing import Optional

import polars as pl
import typer
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# Suppress RDKit warnings
RDLogger.logger().setLevel(RDLogger.CRITICAL)

app = typer.Typer(no_args_is_help=True, help="Clean HTS data and split by assay format.")

PROGRESS_COLUMNS = (
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("{task.completed}/{task.total}"),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
)

# Configuration Constants
MIN_MW = 180.0
MAX_MW = 900.0

# SMARTS pattern for any atom NOT in the allowed list (Hit Dexter 3 allowed set)
# Allowed: H(1), B(5), C(6), N(7), O(8), F(9), Si(14), P(15), S(16), Cl(17), Se(34), Br(35), I(53)
FORBIDDEN_ATOM_SMARTS = (
    "[!#1&!#5&!#6&!#7&!#8&!#9&!#14&!#15&!#16&!#17&!#34&!#35&!#53]"
)


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
    """Stateful processor to hold RDKit objects and pre-compiled patterns."""

    def __init__(self, skip_tautomers: bool = False) -> None:
        self.uncharger = rdMolStandardize.Uncharger()
        self.fragment_chooser = rdMolStandardize.LargestFragmentChooser()
        self.skip_tautomers = skip_tautomers

        # Only initialize the enumerator if we intend to use it
        if not self.skip_tautomers:
            self.tautomer_enumerator = rdMolStandardize.TautomerEnumerator()

        # Compile SMARTS once per worker
        self.forbidden_pattern = Chem.MolFromSmarts(FORBIDDEN_ATOM_SMARTS)

    def process_smiles(self, smi: str | None) -> tuple[str | None, str]:
        """Apply filters and standardization in the most efficient order."""
        if not smi:
            return None, "missing_smiles"

        try:
            mol = Chem.MolFromSmiles(smi)
        except Exception:
            return None, "parse_exception"

        if mol is None:
            return None, "invalid_smiles"

        try:
            mol = self.uncharger.uncharge(mol)
            mol = self.fragment_chooser.choose(mol)
            mol = self.uncharger.uncharge(mol)
        except Exception:
            return None, "uncharger_error"

        try:
            mw = Descriptors.ExactMolWt(mol)
        except Exception:
            return None, "mw_compute_error"

        if not (MIN_MW <= mw <= MAX_MW):
            return None, "molecular_weight_filter"

        try:
            if mol.HasSubstructMatch(self.forbidden_pattern):
                return None, "forbidden_atom"
        except Exception:
            return None, "atom_filter_error"

        if not self.skip_tautomers:
            try:
                mol = self.tautomer_enumerator.Canonicalize(mol)
            except Exception:
                return None, "tautomer_error"

            if mol is None:
                return None, "tautomer_error"

        try:
            canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        except Exception:
            return None, "canonicalization_error"

        if not canonical:
            return None, "canonicalization_error"

        try:
            if Chem.MolFromSmiles(canonical) is None:
                return None, "canonical_validation_failed"
        except Exception:
            return None, "canonical_validation_failed"

        return canonical, "retained"


# Global variable to hold the processor instance in each worker process
_worker_processor: Optional[ChemicalProcessor] = None

def _init_worker(skip_tautomers: bool) -> None:
    """Initialize the RDKit processor once per worker process."""
    global _worker_processor
    _worker_processor = ChemicalProcessor(skip_tautomers=skip_tautomers)

def _process_wrapper(smi: str) -> tuple[Optional[str], str]:
    """Lightweight wrapper to invoke the global processor."""
    if _worker_processor:
        return _worker_processor.process_smiles(smi)
    return None, "missing_processor"


def read_table(path: Path) -> pl.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        return pl.read_csv(path)
    if ext == ".parquet":
        return pl.read_parquet(path)
    raise ValueError(f"Unsupported input extension '{ext}'. Use .csv or .parquet")


def scan_table(path: Path) -> pl.LazyFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        return pl.scan_csv(path)
    if ext == ".parquet":
        return pl.scan_parquet(path)
    raise ValueError(f"Unsupported input extension '{ext}'. Use .csv or .parquet")


def write_table(df: pl.DataFrame, path: Path) -> None:
    ext = path.suffix.lower()
    if ext == ".csv":
        df.write_csv(path)
    elif ext == ".parquet":
        df.write_parquet(path)
    else:
        raise ValueError(f"Unsupported output extension '{ext}'. Use .csv or .parquet")


def sink_table(df: pl.DataFrame | pl.LazyFrame, path: Path) -> None:
    ext = path.suffix.lower()
    if isinstance(df, pl.LazyFrame):
        if ext == ".csv":
            df.sink_csv(path)
        elif ext == ".parquet":
            df.sink_parquet(path)
        else:
            raise ValueError(f"Unsupported output extension '{ext}'. Use .csv or .parquet")
        return

    write_table(df, path)


def rename_cols(
    df: pl.DataFrame | pl.LazyFrame, smiles_col: str, assay_col: str, active_col: str
) -> pl.DataFrame | pl.LazyFrame:
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
    hts_file: Path = typer.Option(..., "--hts-file", help="Path to HTS data file."),
    assay_props_file: Path = typer.Option(..., "--assay-props-file", help="Path to assay properties file."),
    id_to_smiles_file: Path | None = typer.Option(None, "--id-to-smiles-file", help="Optional ID-to-SMILES mapping."),
    id_col: str = typer.Option("compound_id", "--id-col"),
    smiles_col: str = typer.Option("smiles", "--smiles-col"),
    assay_col: str = typer.Option("assay_id", "--assay-col"),
    active_col: str = typer.Option("active", "--active-col"),
    assay_format_col: str = typer.Option("assay_format", "--assay-format-col"),
    biochemical_format: str = typer.Option("biochemical", "--biochemical-format"),
    cellular_format: str = typer.Option("cellular", "--cellular-format"),
    score_col: str | None = typer.Option(None, "--score-col"),
    score_threshold: float = typer.Option(3.0, "--score-threshold"),
    biochemical_out: Path = typer.Option(..., "--biochemical-out"),
    cellular_out: Path = typer.Option(..., "--cellular-out"),
    rename_columns: bool = typer.Option(False, "--rename-cols"),
    skip_tautomers: bool = typer.Option(False, "--skip-tautomers", help="Skip expensive tautomer canonicalization if input is already clean."),
    log_file: Path | None = typer.Option(None, "--log-file"),
    stats_out: Path | None = typer.Option(None, "--stats-out", help="Optional JSON file to write cleaning statistics. Defaults to <biochemical_out>/assay_cleaning_stats.json."),
    n_jobs: int = typer.Option(-1, "--n-jobs", help="Number of CPU cores. -1 for all."),
) -> None:
    """Clean HTS data and split into biochemical and cellular subsets."""
    logger = _setup_logger(log_file)

    if n_jobs < 1:
        n_jobs = multiprocessing.cpu_count()
    
    logger.info("Starting processing with %d CPU cores", n_jobs)
    if skip_tautomers:
        logger.info("Skipping tautomer enumeration (Fast Mode)")

    # 1. Load Data
    hts_lazy = scan_table(hts_file)
    assay_props_lazy = scan_table(assay_props_file)

    # 2. Prepare SMILES List
    if id_to_smiles_file:
        id_to_smiles_df = (
            scan_table(id_to_smiles_file)
            .select([id_col, smiles_col])
            .collect()
        )
    else:
        id_to_smiles_df = (
            hts_lazy
            .select([id_col, smiles_col])
            .unique()
            .collect(engine="streaming")
        )

    id_to_smiles_df = id_to_smiles_df.filter(pl.col(smiles_col).is_not_null())
    
    original_smiles_count = id_to_smiles_df.height
    smiles_list = id_to_smiles_df[smiles_col].to_list()

    # 3. Parallel Processing
    canonical_results = []
    drop_reasons = []
    with Progress(*PROGRESS_COLUMNS) as progress:
        task_id = progress.add_task("Processing structures", total=len(smiles_list))

        # Initialize workers with the configuration flag
        with multiprocessing.Pool(
            processes=n_jobs, 
            initializer=_init_worker, 
            initargs=(skip_tautomers,)
        ) as pool:
            
            results_iter = pool.imap(_process_wrapper, smiles_list, chunksize=1000)
            
            for res in results_iter:
                canonical, reason = res
                canonical_results.append(canonical)
                drop_reasons.append(reason)
                progress.advance(task_id)

    # 4. Reattach Results
    processed_df = pl.DataFrame(
        {
            smiles_col: smiles_list,
            "canonical_smiles": canonical_results,
            "clean_reason": drop_reasons,
        }
    )

    reason_counts = (
        processed_df.select(pl.col("clean_reason"))
        .to_series()
        .value_counts(sort=True)
        .to_dicts()
    )
    reason_counts_map = {
        entry["clean_reason"]: int(entry["count"]) for entry in reason_counts
    }

    processed_df = processed_df.filter(pl.col("canonical_smiles").is_not_null())
    retained_smiles_count = processed_df.height

    logger.info("Structure cleaning stats:")
    logger.info("  - Input unique structures: %s", f"{original_smiles_count:,}")
    logger.info("  - Valid canonical structures: %s", f"{retained_smiles_count:,}")
    logger.info("  - Dropped: %s", f"{original_smiles_count - retained_smiles_count:,}")
    logger.info("Drop reasons (unique SMILES):")
    for reason, count in sorted(reason_counts_map.items(), key=lambda kv: kv[1], reverse=True):
        logger.info("  - %s: %s", reason, f"{count:,}")

    # 5. Join Back to Data
    clean_ids = (
        id_to_smiles_df
        .join(processed_df, on=smiles_col, how="inner")
        .select([id_col, "canonical_smiles"])
    )

    clean_hts_lazy = (
        hts_lazy.join(clean_ids.lazy(), on=id_col, how="inner")
        .with_columns(pl.col("canonical_smiles").alias(smiles_col))
        .drop("canonical_smiles")
    )

    # 6. Join Assay Props
    clean_hts_lazy = clean_hts_lazy.join(
        assay_props_lazy.select([assay_col, assay_format_col]),
        on=assay_col,
        how="inner"
    )

    if rename_columns:
        clean_hts_lazy = rename_cols(clean_hts_lazy, smiles_col, assay_col, active_col)

    # 7. Activity Logic
    cols = clean_hts_lazy.collect_schema().names()
    if "active" not in cols:
        if score_col and score_col in cols:
            logger.info("Deriving 'active' from '%s' >= %s", score_col, score_threshold)
            clean_hts_lazy = clean_hts_lazy.with_columns(
                (pl.col(score_col).abs() >= score_threshold).cast(pl.Int8).alias("active")
            )
        else:
            raise typer.BadParameter("Missing 'active' column or valid --score-col.")
    else:
        clean_hts_lazy = clean_hts_lazy.with_columns(pl.col("active").cast(pl.Int8))

    compound_col_out = "compound_id" if rename_columns else id_col
    assay_col_out = "assay_id" if rename_columns else assay_col

    def summarize_lazy(
        lf: pl.LazyFrame, compound_column: str, assay_column: str
    ) -> dict[str, int]:
        stats = (
            lf.select(
                pl.len().alias("rows"),
                pl.col(compound_column).n_unique().alias("unique_compounds"),
                pl.col(assay_column).n_unique().alias("unique_assays"),
            )
            .collect(streaming=True)
            .to_dicts()[0]
        )
        return {k: int(v) for k, v in stats.items()}

    input_summary = summarize_lazy(hts_lazy, id_col, assay_col)
    logger.info("Input summary: %s", input_summary)

    # 8. Split and Write
    biochemical_df = clean_hts_lazy.filter(pl.col(assay_format_col) == biochemical_format)
    cellular_df = clean_hts_lazy.filter(pl.col(assay_format_col) == cellular_format)

    logger.info("Writing output...")
    sink_table(biochemical_df, biochemical_out)
    sink_table(cellular_df, cellular_out)
    logger.info("Done.")

    output_summary = {
        "combined_clean": summarize_lazy(clean_hts_lazy, compound_col_out, assay_col_out),
        "biochemical": summarize_lazy(biochemical_df, compound_col_out, assay_col_out),
        "cellular": summarize_lazy(cellular_df, compound_col_out, assay_col_out),
    }

    logger.info("Output summary: %s", output_summary)

    if stats_out is None:
        stats_out = biochemical_out.parent / "assay_cleaning_stats.json"

    stats_out.parent.mkdir(parents=True, exist_ok=True)
    stats_payload = {
        "input": input_summary,
        "structure_cleaning": {
            "unique_smiles": original_smiles_count,
            "valid_canonical_smiles": retained_smiles_count,
            "dropped": original_smiles_count - retained_smiles_count,
            "drop_reasons": reason_counts_map,
            "skip_tautomers": skip_tautomers,
        },
        "output": output_summary,
    }
    stats_out.write_text(json.dumps(stats_payload, indent=2))
    logger.info("Wrote stats to %s", stats_out)

def main() -> None:
    app()


if __name__ == "__main__":
    main()
