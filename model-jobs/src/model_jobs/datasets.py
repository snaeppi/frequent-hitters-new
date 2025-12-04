from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Sequence

import polars as pl

logger = logging.getLogger(__name__)


def _ensure_parent_dir(path: Path) -> None:
    if parent := path.parent:
        parent.mkdir(parents=True, exist_ok=True)


def _assert_columns(path: Path, required: Iterable[str]) -> None:
    """Raise ``ValueError`` if any required column is absent."""
    required_set = set(required)
    if not required_set:
        return

    schema = pl.read_parquet(path, n_rows=0)
    missing = required_set.difference(schema.columns)
    if missing:
        cols = ", ".join(sorted(missing))
        raise ValueError(f"{path} missing required columns: {cols}")


def _sink_parquet(
    lf: pl.LazyFrame,
    output_path: Path,
    columns: Sequence[str],
) -> None:
    """Materialise a lazy frame to Parquet selecting the requested columns."""
    _ensure_parent_dir(output_path)
    lf.select([pl.col(name) for name in columns]).sink_parquet(str(output_path))


def _apply_compound_min_screens(
    lf: pl.LazyFrame,
    *,
    min_screens: float | None,
    screens_column: str,
) -> pl.LazyFrame:
    """Filter compounds by minimum screen count when requested."""
    if min_screens is None:
        return lf
    return lf.filter(pl.col(screens_column) >= float(min_screens))


def write_regression_dataset(
    input_path: str | Path,
    output_path: str | Path,
    *,
    smiles_column: str = "smiles",
    split_column: str = "split",
    target_columns: Sequence[str] | None = None,
    target_column: str = "score",
    compound_min_screens: float | None = None,
    compound_screens_column: str = "screens",
) -> Path:
    """Prepare the regression dataset expected by Chemprop."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    targets = list(dict.fromkeys(target_columns or [target_column]))
    required = {smiles_column, split_column, compound_screens_column, *targets}
    _assert_columns(input_path, required)

    logger.info("Creating regression dataset from %s -> %s", input_path, output_path)
    lf = pl.scan_parquet(input_path)
    lf = _apply_compound_min_screens(
        lf,
        min_screens=compound_min_screens,
        screens_column=compound_screens_column,
    )
    columns = [smiles_column, split_column, compound_screens_column, *targets]
    _sink_parquet(lf, output_path, columns)
    return output_path


def write_trimmed_dataset(
    input_path: str | Path,
    output_path: str | Path,
    *,
    drop_columns: Sequence[str],
    keep_numeric_columns: bool = False,
    keep_columns: Sequence[str] | None = None,
) -> Path:
    """Drop columns to reduce memory footprint.

    When ``keep_numeric_columns`` or ``keep_columns`` are provided the function
    switches to a whitelist mode, retaining only digit-only column names and/or
    the explicitly listed columns (after removing any duplicates from
    ``drop_columns``). Otherwise it behaves like a simple drop helper and keeps
    every column not mentioned in ``drop_columns``.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    schema = pl.read_parquet(input_path, n_rows=0)
    all_columns = schema.columns
    existing_columns = set(all_columns)
    drop_set = {col for col in drop_columns if col in existing_columns}
    missing = [col for col in drop_columns if col not in existing_columns]
    if missing:
        logger.debug("Columns not present in %s (skipped): %s", input_path, ", ".join(missing))

    keep_set = set(keep_columns) if keep_columns else set()
    drop_set.difference_update(keep_set)

    numeric_set = {col for col in all_columns if col.isdigit()} if keep_numeric_columns else set()
    keep_mode = bool(keep_set or numeric_set)
    allowed = keep_set.union(numeric_set) if keep_mode else None

    final_keep: list[str] = []
    for column in all_columns:
        if column in drop_set:
            continue
        if keep_mode and allowed is not None and column not in allowed:
            continue
        final_keep.append(column)

    if not final_keep:
        raise ValueError("Dropping all columns would result in an empty dataset.")

    allowed_desc = (
        f"{len(allowed)} columns (numeric={'yes' if numeric_set else 'no'}, explicit={len(keep_set)})"
        if allowed
        else "all non-dropped columns"
    )
    logger.info(
        "Creating trimmed dataset from %s -> %s (dropped: %s, allowed: %s)",
        input_path,
        output_path,
        ", ".join(sorted(drop_set)) if drop_set else "none",
        allowed_desc,
    )
    lf = pl.scan_parquet(input_path).select([pl.col(name) for name in final_keep])
    _ensure_parent_dir(output_path)
    lf.sink_parquet(str(output_path))
    return output_path


def write_threshold_classifier_dataset(
    input_path: str | Path,
    output_path: str | Path,
    *,
    smiles_column: str = "smiles",
    split_column: str = "split",
    metric_column: str = "score",
    target_column: str = "target",
    lower_threshold: float,
    upper_threshold: float,
    compound_min_screens: float | None = None,
    compound_screens_column: str = "screens",
) -> Path:
    """Prepare a binary classification dataset by thresholding a continuous metric."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    if lower_threshold > upper_threshold:
        raise ValueError("lower_threshold must be less than or equal to upper_threshold")

    required = {smiles_column, split_column, metric_column, compound_screens_column}
    _assert_columns(input_path, required)

    logger.info(
        "Creating threshold classifier dataset using %s <= %.6f / >= %.6f from %s -> %s",
        metric_column,
        lower_threshold,
        upper_threshold,
        input_path,
        output_path,
    )

    metric = pl.col(metric_column)
    target_expr = (
        pl.when(metric >= upper_threshold)
        .then(pl.lit(1, dtype=pl.Int8))
        .when(metric <= lower_threshold)
        .then(pl.lit(0, dtype=pl.Int8))
        .otherwise(pl.lit(None, dtype=pl.Int8))
        .alias(target_column)
    )

    lf = pl.scan_parquet(input_path)
    lf = _apply_compound_min_screens(
        lf,
        min_screens=compound_min_screens,
        screens_column=compound_screens_column,
    )
    lf = (
        lf.select(
            [
                pl.col(smiles_column),
                pl.col(split_column),
                metric,
                pl.col(compound_screens_column),
            ]
        )
        .filter(metric.is_not_null())
        .with_columns(target_expr)
        .filter(pl.col(target_column).is_not_null())
        .select(
            [
                pl.col(smiles_column),
                pl.col(split_column),
                pl.col(target_column).cast(pl.Int8),
                pl.col(compound_screens_column),
            ]
        )
    )

    _ensure_parent_dir(output_path)
    lf.sink_parquet(str(output_path))
    return output_path


def write_smiles_csv(
    input_path: str | Path,
    output_path: str | Path,
    *,
    smiles_column: str = "smiles",
    filters: dict[str, Sequence[object]] | None = None,
    compound_min_screens: float | None = None,
    compound_screens_column: str = "screens",
) -> Path:
    """Extract SMILES strings (optionally filtered) for Chemprop prediction."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    required = {smiles_column}
    if filters:
        required.update(filters.keys())
    if compound_min_screens is not None:
        required.add(compound_screens_column)
    _assert_columns(input_path, required)
    logger.info("Extracting SMILES from %s -> %s", input_path, output_path)

    lf = pl.scan_parquet(input_path)
    if filters:
        for column, values in filters.items():
            if not values:
                continue
            lf = lf.filter(pl.col(column).is_in(list(values)))

    lf = _apply_compound_min_screens(
        lf,
        min_screens=compound_min_screens,
        screens_column=compound_screens_column,
    )
    lf = lf.select(pl.col(smiles_column))

    _ensure_parent_dir(output_path)
    lf.sink_csv(str(output_path))
    return output_path


def _load_threshold_mapping(json_path: Path) -> dict[float, float]:
    """
    Load a mapping from percentile -> threshold value from a JSON file.

    We only care about "percentiles". Supported shapes:
      - {"percentiles": {"50": 0.047, "55": 0.052, ...}}
      - {"percentiles": [{"percentile": 50, "hit_rate": 0.047}, ...]}
      - Direct mapping as a fallback: {"50": 0.047, "55": 0.052, ...}
    """
    import json

    def _extract_value(val: object) -> float | None:
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, dict):
            if "hit_rate" in val and isinstance(val["hit_rate"], (int, float)):
                return float(val["hit_rate"])
            if "value" in val and isinstance(val["value"], (int, float)):
                return float(val["value"])
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Focus on "percentiles" if present; otherwise attempt to interpret the object directly.
    obj = data["percentiles"] if isinstance(data, dict) and "percentiles" in data else data

    # Case A: dict mapping. Support both direct mappings like {"50": 0.047, ...}
    # and nested payloads like {"score": {"50": 0.047, ...}}.
    if isinstance(obj, dict):
        mapping_dict: dict[float, float] = {}

        # Nested shape: {"metric": {"50": 0.047, ...}, ...}
        for outer_val in obj.values():
            if not isinstance(outer_val, dict):
                continue
            for k, v in outer_val.items():
                try:
                    p = float(k)
                except (TypeError, ValueError):
                    continue
                val = _extract_value(v)
                if val is not None:
                    mapping_dict[p] = val

        # Flat shape: {"50": 0.047, ...}
        if not mapping_dict:
            for k, v in obj.items():
                try:
                    p = float(k)
                except (TypeError, ValueError):
                    # skip non-numeric keys (e.g., assay metadata)
                    continue
                val = _extract_value(v)
                if val is not None:
                    mapping_dict[p] = val

        if mapping_dict:
            return mapping_dict

    # Case B: list of records like [{"percentile": 50, "hit_rate": 0.047}, ...]
    if isinstance(obj, list):
        mapping_list: dict[float, float] = {}
        for el in obj:
            if not isinstance(el, dict) or "percentile" not in el:
                continue
            try:
                p = float(el["percentile"])
            except (TypeError, ValueError):
                continue
            val = _extract_value(el)
            if val is not None:
                mapping_list[p] = val
        if mapping_list:
            return mapping_list

    raise ValueError(f"Could not interpret percentile mapping in {json_path}")


def resolve_thresholds(
    *,
    lower_threshold: float | None,
    upper_threshold: float | None,
    thresholds_json: str | Path | None,
    lower_percentile: float | None,
    upper_percentile: float | None,
) -> tuple[float, float]:
    # Case 1: explicit thresholds provided
    if lower_threshold is not None or upper_threshold is not None:
        if lower_threshold is None or upper_threshold is None:
            raise ValueError("Both lower_threshold and upper_threshold must be provided together.")
        return float(lower_threshold), float(upper_threshold)

    # Case 2: derive from JSON mapping of percentiles
    if thresholds_json is None:
        raise ValueError("Must provide either explicit thresholds or a thresholds_json with percentiles.")
    if lower_percentile is None or upper_percentile is None:
        raise ValueError("Both lower_percentile and upper_percentile are required with thresholds_json.")

    mapping = _load_threshold_mapping(Path(thresholds_json))
    if not mapping:
        raise ValueError(f"No valid percentile mapping found in {thresholds_json}")

    def lookup(p: float) -> float:
        if p in mapping:
            return mapping[p]
        for k, v in mapping.items():
            if abs(k - p) < 1e-6:
                return v
        raise ValueError(
            f"Percentile {p} not found in {thresholds_json}. Available: {sorted(mapping.keys())}"
        )

    return lookup(float(lower_percentile)), lookup(float(upper_percentile))
