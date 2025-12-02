"""Hydra entrypoint for the pipeline package."""

from __future__ import annotations

import copy
import shutil
from pathlib import Path

import hydra
import polars as pl
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from . import eb, export, filters, io, manifest, split


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


def run_pipeline(cfg: DictConfig, *, resolved_cfg_root: dict, hydra_output_dir: Path) -> None:
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

        data_lf, qc_counts = io.load_assay_table(input_path, assay_format)
        (
            filtered_lf,
            assay_metadata,
            retained_assays,
            excluded_assays,
            assay_filter_stats,
        ) = filters.apply_assay_filters(
            data_lf,
            output_dir,
            std_k=float(cfg.filters.assay_hit_rate_outlier_std_k),
            min_screens_per_assay=int(cfg.filters.min_screens_per_assay),
            enable_plots=bool(cfg.enable_plots),
        )

        assays_after_hit_rate = assay_metadata.filter(
            pl.col("excluded_due_to_hit_rate").not_()
        )

        _reliable_df, all_compounds_df, eb_summary = eb.build_compound_metadata(
            filtered_lf,
            output_dir,
            min_screens_per_compound=int(cfg.filters.min_screens_per_compound),
            min_screens_for_prior_fit=int(cfg.filters.min_screens_for_prior_fit),
            enable_plots=bool(cfg.enable_plots),
        )

        fractions = {split_name: float(value) for split_name, value in cfg.split.fractions.items()}
        split_map_df, split_summary = split.assign_scaffold_splits(
            all_compounds_df,
            output_dir,
            fractions=fractions,
            seed=int(cfg.split.seed),
            enable_plots=bool(cfg.enable_plots),
            scaffold_store=scaffold_store,
        )

        dataset_counts, thresholds = export.write_model_datasets(
            assay_format,
            filtered_lf,
            all_compounds_df,
            split_map_df,
            output_dir,
            filter_threshold=eb_summary["filter_threshold"],
        )

        assay_counts = {
            "total": int(assay_metadata.height),
            "retained": int(retained_assays.height),
            "excluded": int(excluded_assays.height),
            "after_hit_rate_filter": int(assays_after_hit_rate.height),
        }

        format_config = copy.deepcopy(resolved_cfg_root)
        if isinstance(format_config, dict):
            format_config["assay_format"] = assay_format

        manifest.write_manifest(
            output_dir,
            assay_format=assay_format,
            resolved_config=format_config,
            qc=qc_counts,
            assay_counts=assay_counts,
            filter_stats={
                **assay_filter_stats,
                "compound_filter_mode": eb_summary["filter_mode"],
                "compound_filter_threshold": eb_summary["filter_threshold"],
                "min_screens_per_compound": int(cfg.filters.min_screens_per_compound),
                "min_screens_for_prior_fit": int(cfg.filters.min_screens_for_prior_fit),
            },
            eb_summary=eb_summary,
            split_summary=split_summary,
            export_counts=dataset_counts,
            thresholds=thresholds,
            split_seed=int(cfg.split.seed),
        )

        hydra_config_dir = hydra_output_dir / ".hydra"
        if hydra_config_dir.exists():
            shutil.copytree(hydra_config_dir, output_dir / ".hydra", dirs_exist_ok=True)
            resolved_config_path = hydra_config_dir / "config.yaml"
            if resolved_config_path.exists():
                shutil.copy2(resolved_config_path, output_dir / "resolved_config.yaml")
            overrides_path = hydra_config_dir / "overrides.yaml"
            if overrides_path.exists():
                shutil.copy2(overrides_path, output_dir / "overrides.yaml")

    if not scaffold_store.is_empty() or scaffold_path.exists():
        scaffold_store.save(scaffold_path)


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    resolved_cfg_root = OmegaConf.to_container(cfg, resolve=True)
    hydra_output_dir = Path(HydraConfig.get().runtime.output_dir)
    run_pipeline(cfg, resolved_cfg_root=resolved_cfg_root, hydra_output_dir=hydra_output_dir)


if __name__ == "__main__":
    main()
