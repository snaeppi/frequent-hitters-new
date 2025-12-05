"""Hydra entrypoint for the frequent hitter data pipeline."""

from __future__ import annotations

import shutil
from pathlib import Path

import hydra
import polars as pl
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from . import eb, export, filters, io, split

PROGRESS_COLUMNS = (
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("{task.completed}/{task.total}"),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
)


def _selected_formats(assay_format: str) -> list[str]:
    assay_format = assay_format.lower()
    if assay_format == "both":
        return ["biochemical", "cellular"]
    if assay_format not in {"biochemical", "cellular"}:
        raise ValueError("assay_format must be one of: biochemical | cellular | both")
    return [assay_format]


def _float_or_none(value) -> float | None:
    if value is None:
        return None
    return float(value)


def _resolve_input_path(value) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none"}:
        return None
    return Path(to_absolute_path(text))


def run_pipeline(cfg: DictConfig, *, hydra_output_dir: Path) -> None:
    output_root = Path(to_absolute_path(cfg.paths.output_root))
    input_mapping = OmegaConf.to_container(cfg.paths.input, resolve=True) or {}
    if not isinstance(input_mapping, dict):
        raise TypeError("paths.input must be a mapping of assay_format -> filepath.")

    selected_formats = _selected_formats(cfg.assay_format)
    resolved_inputs: list[tuple[str, Path]] = []
    for assay_format in selected_formats:
        input_path = _resolve_input_path(input_mapping.get(assay_format))
        if input_path is None:
            if cfg.assay_format != "both":
                raise ValueError(
                    "paths.input."
                    f"{assay_format} must be provided when assay_format='{cfg.assay_format}'."
                )
            continue
        resolved_inputs.append((assay_format, input_path))

    if not resolved_inputs:
        raise ValueError(
            "No assay inputs supplied. Provide paths.input.biochemical and/or paths.input.cellular."
        )

    scaffolds_value = cfg.paths.get("scaffolds", None) if hasattr(cfg.paths, "get") else None
    scaffold_path = _resolve_input_path(scaffolds_value)
    if scaffold_path is None:
        scaffold_path = output_root / "scaffold_assignments.parquet"
        scaffold_store = split.ScaffoldStore()
    else:
        if scaffold_path.exists():
            scaffold_store = split.ScaffoldStore.from_file(scaffold_path)
        else:
            scaffold_store = split.ScaffoldStore()

    for assay_format, input_path in resolved_inputs:
        output_dir = output_root / assay_format
        output_dir.mkdir(parents=True, exist_ok=True)

        task_label = assay_format.capitalize()
        total_steps = 6
        with Progress(*PROGRESS_COLUMNS) as progress:
            task_id = progress.add_task(f"{task_label}: preparing", total=total_steps)

            progress.update(task_id, description=f"{task_label}: load assay table")
            data_lf, _qc_counts = io.load_assay_table(input_path, assay_format)
            progress.advance(task_id)

            progress.update(task_id, description=f"{task_label}: assay filtering")
            (
                filtered_lf,
                _assay_metadata,
                _retained_assays,
                _excluded_assays,
                _assay_filter_stats,
            ) = filters.apply_assay_filters(
                data_lf,
                output_dir,
                std_k=float(cfg.filters.assay_hit_rate_outlier_std_k),
                min_screens_per_assay=int(cfg.filters.min_screens_per_assay),
                enable_plots=bool(cfg.enable_plots),
            )
            progress.advance(task_id)

            progress.update(task_id, description=f"{task_label}: reliability scoring")
            _reliable_df, all_compounds_df, eb_summary = eb.build_compound_metadata(
                filtered_lf,
                output_dir,
                min_screens_per_compound=int(cfg.filters.min_screens_per_compound),
                min_screens_for_prior_fit=int(cfg.filters.min_screens_for_prior_fit),
                enable_plots=bool(cfg.enable_plots),
            )
            progress.advance(task_id)

            progress.update(task_id, description=f"{task_label}: scaffold splits")
            fractions = {split_name: float(value) for split_name, value in cfg.split.fractions.items()}
            split_map_df, _split_summary = split.assign_scaffold_splits(
                all_compounds_df,
                output_dir,
                fractions=fractions,
                seed=int(cfg.split.seed),
                enable_plots=bool(cfg.enable_plots),
                scaffold_store=scaffold_store,
            )
            progress.advance(task_id)

            progress.update(task_id, description=f"{task_label}: export datasets")
            _dataset_counts, _thresholds = export.write_model_datasets(
                assay_format,
                filtered_lf,
                all_compounds_df,
                split_map_df,
                output_dir,
                filter_threshold=eb_summary["filter_threshold"],
            )
            progress.advance(task_id)

            progress.update(task_id, description=f"{task_label}: copy Hydra configs")
            hydra_config_dir = hydra_output_dir / ".hydra"
            if hydra_config_dir.exists():
                shutil.copytree(hydra_config_dir, output_dir / ".hydra", dirs_exist_ok=True)
                resolved_config_path = hydra_config_dir / "config.yaml"
                if resolved_config_path.exists():
                    shutil.copy2(resolved_config_path, output_dir / "resolved_config.yaml")
                overrides_path = hydra_config_dir / "overrides.yaml"
                if overrides_path.exists():
                    shutil.copy2(overrides_path, output_dir / "overrides.yaml")
            progress.advance(task_id)
            progress.update(task_id, description=f"[green]{task_label}: complete")

    if not scaffold_store.is_empty() or scaffold_path.exists():
        scaffold_store.save(scaffold_path)


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    hydra_output_dir = Path(HydraConfig.get().runtime.output_dir)
    run_pipeline(cfg, hydra_output_dir=hydra_output_dir)


if __name__ == "__main__":
    main()
