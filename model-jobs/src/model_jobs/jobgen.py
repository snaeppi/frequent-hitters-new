from __future__ import annotations

import re
import shlex
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol, TypeVar

import yaml

__all__ = ["ConfigError", "load_submission_plan", "JobDefinition", "GlobalConfig"]


class ConfigError(Exception):
    """Raised when the submission configuration file is invalid."""


class BaseGlobalConfig(Protocol):
    models_dir: Path
    temp_dir: Path
    cpus: int
    conda_commands: list[str]
    conda_activate: str
    python_executable: str
    epochs: int
    seed: int
    metrics_classification: list[str]
    metrics_regression: list[str]
    chemprop_train_cmd: str
    submit: bool
    dataset_aliases: dict[str, Path]
    base_path: Path


GC = TypeVar("GC", bound=BaseGlobalConfig)


@dataclass
class Renderers:
    header: Callable[[str, GC], str]
    common_env: Callable[[GC, str], str]


@dataclass
class JobDefinition:
    job_name: str
    filename: str
    content: str
    submit: bool


def _load_submission_plan(
    config_path: Path,
    *,
    parse_global_config: Callable[[dict, Path], GC],
    renderers: Renderers,
) -> tuple[GC, list[JobDefinition]]:
    config_path = config_path.resolve()
    try:
        raw = yaml.safe_load(config_path.read_text())
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration file not found: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Unable to parse YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError("Configuration must be a mapping.")

    base_path = config_path.parent
    global_cfg = parse_global_config(raw.get("global", {}), base_path)
    tasks_raw = raw.get("tasks")
    if not isinstance(tasks_raw, list) or not tasks_raw:
        raise ConfigError("The configuration must include a non-empty 'tasks' list.")

    jobs: list[JobDefinition] = []
    for entry in tasks_raw:
        if not isinstance(entry, dict):
            raise ConfigError("Each task entry must be a mapping.")

        task_type = entry.get("type")
        if not task_type:
            raise ConfigError("Task is missing required 'type' field.")
        task_type_normalized = task_type.replace("-", "_").lower()

        if task_type_normalized in {"multilabel", "multi_label"}:
            job_defs = _build_multilabel_jobs(entry, global_cfg, renderers)
        elif task_type_normalized == "regression":
            job_defs = _build_regression_jobs(entry, global_cfg, renderers)
        elif task_type_normalized in {"threshold", "threshold_classification"}:
            job_defs = _build_threshold_jobs(entry, global_cfg, renderers)
        elif task_type_normalized in {"predict", "prediction"}:
            raise ConfigError("Prediction-only tasks are no longer supported; include splits in the training dataset instead.")
        else:
            raise ConfigError(f"Unknown task type '{task_type}'.")

        jobs.extend(job_defs)

    return global_cfg, jobs


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def _resolve_dataset_path(
    value: str | Path | None, global_cfg: BaseGlobalConfig, field_name: str
) -> Path:
    if value is None:
        raise ConfigError(f"Missing required path for {field_name}.")
    if isinstance(value, Path):
        return value
    if value in global_cfg.dataset_aliases:
        return global_cfg.dataset_aliases[value]
    return _resolve_path(value, global_cfg.base_path)


def _extract_job_name(task: dict) -> str:
    name = task.get("job_name") or task.get("name")
    if not name:
        raise ConfigError("Task is missing 'job_name' (or 'name').")
    return str(name)


def _slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return slug.strip("_") or "job"


def _format_model_dir(
    raw_value: str | None, global_cfg: BaseGlobalConfig, context: dict[str, str]
) -> Path:
    if raw_value is None:
        return global_cfg.models_dir / context["job_slug"]
    formatted = raw_value.format(**context)
    return _resolve_path(formatted, global_cfg.base_path)


def _split_column_from_task(task: dict, *, default_seed: int) -> str:
    """Return the split column to use for a task, defaulting to the task seed."""
    split_override = task.get("split_column")
    if split_override:
        return str(split_override)
    seed_override = task.get("split_seed")
    seed_value = default_seed if seed_override is None else int(seed_override)
    return f"split{seed_value}"


def _screens_weight_mode_from_task(task: dict) -> str:
    """Normalize the screens weight mode."""
    mode = str(task.get("screens_weight_mode", "none")).strip().lower()
    if mode not in {"none", "linear", "sqrt"}:
        raise ConfigError("screens_weight_mode must be one of: none, linear, sqrt.")
    return mode


# ---------------------------------------------------------------------------
# Job builders
# ---------------------------------------------------------------------------


def _build_multilabel_jobs(task: dict, global_cfg: GC, renderers: Renderers) -> list[JobDefinition]:
    job_name = _extract_job_name(task)
    job_slug = _slugify(job_name)

    trainval_path = _resolve_dataset_path(task.get("trainval_path"), global_cfg, "trainval_path")
    epochs = int(task.get("epochs", global_cfg.epochs))
    seed = int(task.get("seed", global_cfg.seed))
    split_seed_value = int(task.get("split_seed", seed))
    split_column = _split_column_from_task(task, default_seed=seed)
    keep_columns = ["smiles", split_column]
    keep_columns.extend(task.get("extra_keep_columns", []))
    keep_columns = list(dict.fromkeys(map(str, keep_columns)))
    ensemble_size = int(task.get("ensemble_size", 1))

    model_context = {
        "job_name": job_name,
        "job_slug": job_slug,
    }
    model_dir = _format_model_dir(task.get("model_dir"), global_cfg, model_context)

    submit_flag = _should_submit(task, global_cfg)
    content = _render_multilabel_script(
        job_name=job_name,
        job_slug=job_slug,
        data_path=trainval_path,
        model_dir=model_dir,
        keep_columns=keep_columns,
        global_cfg=global_cfg,
        epochs=epochs,
        seed=seed,
        ensemble_size=ensemble_size,
        train_enabled=bool(task.get("train", True)),
        classification_metrics=_task_metrics(
            task, "classification_metrics", global_cfg.metrics_classification
        ),
        split_column=split_column,
        renderers=renderers,
    )
    filename = f"{job_slug}.sh"
    return [
        JobDefinition(job_name=job_name, filename=filename, content=content, submit=submit_flag)
    ]


def _build_regression_jobs(task: dict, global_cfg: GC, renderers: Renderers) -> list[JobDefinition]:
    job_name = _extract_job_name(task)
    job_slug = _slugify(job_name)
    trainval_path = _resolve_dataset_path(task.get("trainval_path"), global_cfg, "trainval_path")
    task_type = str(task.get("task_type", "regression"))
    targets = task.get("targets") or task.get("target_columns")
    if targets:
        targets_list = [str(t) for t in targets]
    else:
        target = task.get("target_column")
        if not target:
            seed = int(task.get("seed", global_cfg.seed))
            split_seed_value = int(task.get("split_seed", seed))
            target = f"score_seed{split_seed_value}"
        targets_list = [str(target)]

    epochs = int(task.get("epochs", global_cfg.epochs))
    seed = int(task.get("seed", global_cfg.seed))
    split_seed_value = int(task.get("split_seed", seed))
    split_column = _split_column_from_task(task, default_seed=seed)
    ensemble_size = int(task.get("ensemble_size", 1))
    compound_min = task.get("compound_min_screens")
    compound_min_value = float(compound_min) if compound_min is not None else None
    compound_screens_column = task.get("compound_screens_column", "screens")
    if compound_screens_column is None:
        compound_screens_column = "screens"
    compound_screens_column = str(compound_screens_column)
    screens_weight_mode = _screens_weight_mode_from_task(task)

    job_defs: list[JobDefinition] = []
    for target in targets_list:
        target_slug = _slugify(target)
        per_job_name = job_name if len(targets_list) == 1 else f"{job_name}_{target_slug}"
        per_slug = _slugify(per_job_name)
        context = {
            "job_name": per_job_name,
            "job_slug": per_slug,
            "target": target_slug,
            "target_column": target,
        }
        model_dir = _format_model_dir(task.get("model_dir"), global_cfg, context)

        submit_flag = _should_submit(task, global_cfg)
        content = _render_regression_script(
            job_name=per_job_name,
            job_slug=per_slug,
            data_path=trainval_path,
            model_dir=model_dir,
            target_column=target,
            task_type=task_type,
            global_cfg=global_cfg,
            epochs=epochs,
            seed=seed,
            ensemble_size=ensemble_size,
            train_enabled=bool(task.get("train", True)),
            regression_metrics=_regression_metrics(task, global_cfg),
            compound_min_screens=compound_min_value,
            compound_screens_column=compound_screens_column,
            split_column=split_column,
            renderers=renderers,
            screens_weight_mode=screens_weight_mode,
        )
        filename = f"{per_slug}.sh"
        job_defs.append(JobDefinition(per_job_name, filename, content, submit_flag))

    return job_defs


def _build_threshold_jobs(task: dict, global_cfg: GC, renderers: Renderers) -> list[JobDefinition]:
    job_name = _extract_job_name(task)
    job_slug = _slugify(job_name)
    trainval_path = _resolve_dataset_path(task.get("trainval_path"), global_cfg, "trainval_path")
    seed = int(task.get("seed", global_cfg.seed))
    split_seed_value = int(task.get("split_seed", seed))
    metric_default = f"score_seed{split_seed_value}"
    metric_column = str(task.get("metric_column", metric_default))
    target_column = str(task.get("target_column", "target"))
    epochs = int(task.get("epochs", global_cfg.epochs))
    split_column = _split_column_from_task(task, default_seed=seed)
    ensemble_size = int(task.get("ensemble_size", 1))
    compound_min = task.get("compound_min_screens")
    compound_min_value = float(compound_min) if compound_min is not None else None
    compound_screens_column = task.get("compound_screens_column", "screens")
    if compound_screens_column is None:
        compound_screens_column = "screens"
    compound_screens_column = str(compound_screens_column)
    screens_weight_mode = _screens_weight_mode_from_task(task)

    threshold_specs = task.get("thresholds")
    iterable = threshold_specs if threshold_specs else [task]

    jobs: list[JobDefinition] = []
    for idx, spec in enumerate(iterable):
        if not isinstance(spec, dict):
            raise ConfigError("thresholds entries must be mappings.")

        lower_percentile = spec.get("lower_percentile")
        upper_percentile = spec.get("upper_percentile")
        lower_threshold = spec.get("lower_threshold")
        upper_threshold = spec.get("upper_threshold")
        if lower_percentile is None and lower_threshold is None:
            raise ConfigError(
                "Threshold specification requires lower_percentile or lower_threshold."
            )
        if upper_percentile is None and upper_threshold is None:
            raise ConfigError(
                "Threshold specification requires upper_percentile or upper_threshold."
            )

        suffix = spec.get("suffix") or spec.get("name_suffix")
        if suffix:
            suffix_str = str(suffix)
        elif lower_percentile is not None and upper_percentile is not None:
            suffix_str = f"p{lower_percentile}_{upper_percentile}"
        elif lower_threshold is not None and upper_threshold is not None:
            suffix_str = f"thr{lower_threshold}_{upper_threshold}"
        else:
            suffix_str = f"combo{idx + 1}"

        per_job_name = f"{job_name}_{suffix_str}"
        per_slug = _slugify(per_job_name)
        context = {
            "job_name": per_job_name,
            "job_slug": per_slug,
            "suffix": suffix_str,
            "metric": metric_column,
        }
        model_dir = _format_model_dir(task.get("model_dir"), global_cfg, context)

        threshold_args = _threshold_arguments(
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
            thresholds_json=spec.get("thresholds_json") or task.get("thresholds_json"),
            global_cfg=global_cfg,
        )

        submit_flag = _should_submit(task, global_cfg)
        content = _render_threshold_script(
            job_name=per_job_name,
            job_slug=per_slug,
            data_path=trainval_path,
            model_dir=model_dir,
            metric_column=metric_column,
            target_column=target_column,
            global_cfg=global_cfg,
            epochs=epochs,
            seed=seed,
            ensemble_size=ensemble_size,
            train_enabled=bool(task.get("train", True)),
            threshold_args=threshold_args,
            classification_metrics=_task_metrics(
                task, "classification_metrics", global_cfg.metrics_classification
            ),
            compound_min_screens=compound_min_value,
            compound_screens_column=compound_screens_column,
            split_column=split_column,
            renderers=renderers,
            screens_weight_mode=screens_weight_mode,
        )
        filename = f"{per_slug}.sh"
        jobs.append(JobDefinition(per_job_name, filename, content, submit_flag))

    return jobs


def _build_prediction_jobs(task: dict, global_cfg: GC, renderers: Renderers) -> list[JobDefinition]:
    raise ConfigError("Prediction-only tasks are not supported; include predictions via splits in the training dataset.")


def _should_submit(task: dict, global_cfg: BaseGlobalConfig) -> bool:
    submit_override = task.get("submit")
    if submit_override is None:
        return global_cfg.submit
    return bool(submit_override)


def _task_metrics(task: dict, key: str, defaults: list[str]) -> list[str]:
    metrics = task.get(key)
    if metrics is None:
        return defaults
    return [str(m) for m in metrics]


def _regression_metrics(task: dict, global_cfg: BaseGlobalConfig) -> list[str]:
    return _task_metrics(task, "regression_metrics", global_cfg.metrics_regression)


def _threshold_arguments(
    *,
    lower_percentile: float | None,
    upper_percentile: float | None,
    lower_threshold: float | None,
    upper_threshold: float | None,
    thresholds_json: str | Path | None,
    global_cfg: BaseGlobalConfig,
) -> list[tuple[str, str]]:
    args: list[tuple[str, str]] = []
    if lower_threshold is not None or upper_threshold is not None:
        if lower_threshold is None or upper_threshold is None:
            raise ConfigError("Both lower_threshold and upper_threshold are required together.")
        args.append(("--lower-threshold", str(lower_threshold)))
        args.append(("--upper-threshold", str(upper_threshold)))
        return args

    if thresholds_json is None:
        raise ConfigError("thresholds_json is required when using percentiles.")
    thresh_path = _resolve_dataset_path(thresholds_json, global_cfg, "thresholds_json")
    args.append(("--thresholds-json", str(thresh_path)))
    if lower_percentile is None or upper_percentile is None:
        raise ConfigError("Both lower_percentile and upper_percentile are required.")
    args.append(("--lower-percentile", str(lower_percentile)))
    args.append(("--upper-percentile", str(upper_percentile)))
    return args


def _render_multilabel_script(
    *,
    job_name: str,
    job_slug: str,
    data_path: Path,
    model_dir: Path,
    keep_columns: list[str],
    global_cfg: BaseGlobalConfig,
    epochs: int,
    seed: int,
    ensemble_size: int,
    train_enabled: bool,
    classification_metrics: list[str],
    split_column: str,
    renderers: Renderers,
) -> str:
    header = renderers.header(job_name, global_cfg)
    env_block = renderers.common_env(global_cfg, job_name)

    temp_parquet = f"${{TEMP_DIR}}/{job_slug}_multilabel_trimmed.parquet"
    unique_keep = list(dict.fromkeys(keep_columns))

    body_lines: list[str] = [
        header,
        "",
        env_block,
        "",
        f'SPLIT_COLUMN="{split_column}"',
        f'DATA_PATH="{data_path}"',
        f'TEMP_PARQUET="{temp_parquet}"',
        "",
        f'MODEL_DIR="{model_dir}"',
        'mkdir -p "${MODEL_DIR}"',
    ]
    body_lines.append('echo "[INFO] [$START] [$(date)] [$$] Trimming multi-task dataset"')

    trim_lines = [
        '"${PYTHON_BIN}" -m model_jobs.cli trim-columns \\',
        '  --input "${DATA_PATH}" \\',
        '  --output "${TEMP_PARQUET}" \\',
    ]
    trim_lines.extend(f'  --keep-column "{col}" \\' for col in unique_keep)
    trim_lines.append("  --keep-numeric-columns")
    body_lines.append("\n".join(trim_lines))

    if train_enabled:
        train_cmd_lines = [
            "${CHEMPROP_TRAIN} \\",
            '  --data-path "${TEMP_PARQUET}" \\',
            '  --save-dir "${MODEL_DIR}" \\',
            "  --task-type classification \\",
            f"  --ensemble-size {ensemble_size} \\",
            '  --smiles-columns "smiles" \\',
            '  --splits-column "${SPLIT_COLUMN}" \\',
            f"  --epochs {epochs} \\",
            "  --show-individual-scores \\",
        ]
        for metric in classification_metrics:
            train_cmd_lines.append(f'  --metrics "{metric}" \\')
        train_cmd_lines.append(f"  --pytorch-seed {seed} \\")
        train_cmd_lines.append(f"  --num-workers {global_cfg.cpus}")

        body_lines.extend(
            [
                "",
                'echo "[INFO] [$START] [$(date)] [$$] Starting Chemprop multi-task training"',
                "",
                "\n".join(train_cmd_lines),
            ]
        )

    body_lines.extend(
        [
            "",
            'rm "${TEMP_PARQUET}" && echo "[INFO] [$START] [$(date)] [$$] Removed temporary file ${TEMP_PARQUET}"',
        ]
    )

    return "\n".join(body_lines).strip() + "\n"


def _render_regression_script(
    *,
    job_name: str,
    job_slug: str,
    data_path: Path,
    model_dir: Path,
    target_column: str,
    task_type: str,
    global_cfg: BaseGlobalConfig,
    epochs: int,
    seed: int,
    ensemble_size: int,
    train_enabled: bool,
    regression_metrics: list[str],
    compound_min_screens: float | None,
    compound_screens_column: str,
    split_column: str,
    renderers: Renderers,
    screens_weight_mode: str,
) -> str:
    header = renderers.header(job_name, global_cfg)
    env_block = renderers.common_env(global_cfg, job_name)
    temp_parquet = f"${{TEMP_DIR}}/{job_slug}_regression.parquet"
    extra_prep_lines: list[str] = []
    if compound_min_screens is not None:
        min_value = format(compound_min_screens, "g")
        extra_prep_lines.append(f"  --compound-min-screens {min_value}")
    if compound_min_screens is not None or screens_weight_mode != "none":
        extra_prep_lines.append(f'  --compound-screens-column "{compound_screens_column}"')
    if screens_weight_mode != "none":
        extra_prep_lines.append(f'  --screens-weight-mode "{screens_weight_mode}"')

    lines = [
        header,
        "",
        env_block,
        "",
        f'MODEL_DIR="{model_dir}"',
        'mkdir -p "${MODEL_DIR}"',
        f'DATA_PATH="{data_path}"',
        f'TEMP_PARQUET="{temp_parquet}"',
        f'SPLIT_COLUMN="{split_column}"',
        "",
    ]

    lines.append('echo "[INFO] [$START] [$(date)] [$$] Preparing regression dataset"')
    prep_lines = [
        '"${PYTHON_BIN}" -m model_jobs.cli prepare-regression \\',
        '  --input "${DATA_PATH}" \\',
        '  --output "${TEMP_PARQUET}" \\',
        '  --smiles-column "smiles" \\',
        '  --split-column "${SPLIT_COLUMN}" \\',
        f'  --target-column "{target_column}"',
    ]
    if extra_prep_lines:
        prep_lines[-1] += " \\"
        for i, line in enumerate(extra_prep_lines):
            if i < len(extra_prep_lines) - 1:
                prep_lines.append(f"{line} \\")
            else:
                prep_lines.append(line)
    lines.append("\n".join(prep_lines))

    if train_enabled:
        train_cmd_lines = [
            "${CHEMPROP_TRAIN} \\",
            '  --data-path "${TEMP_PARQUET}" \\',
            '  --save-dir "${MODEL_DIR}" \\',
            f"  --task-type {task_type} \\",
            f"  --ensemble-size {ensemble_size} \\",
            '  --smiles-columns "smiles" \\',
            f'  --target-columns "{target_column}" \\',
            '  --splits-column "${SPLIT_COLUMN}" \\',
        ]
        if screens_weight_mode != "none":
            train_cmd_lines.append(f'  --weight-column "{compound_screens_column}" \\')
        train_cmd_lines.extend(
            [
                f"  --epochs {epochs} \\",
                "  --show-individual-scores \\",
            ]
        )
        for metric in regression_metrics:
            train_cmd_lines.append(f'  --metrics "{metric}" \\')
        train_cmd_lines.append(f"  --pytorch-seed {seed} \\")
        train_cmd_lines.append(f"  --num-workers {global_cfg.cpus}")

        lines.extend(
            [
                "",
                'echo "[INFO] [$START] [$(date)] [$$] Starting Chemprop regression training"',
                "",
                "\n".join(train_cmd_lines),
            ]
        )

    lines.extend(
        [
            "",
            'rm "${TEMP_PARQUET}" && echo "[INFO] [$START] [$(date)] [$$] Removed temporary file ${TEMP_PARQUET}"',
        ]
    )

    return "\n".join(lines).strip() + "\n"


def _render_threshold_script(
    *,
    job_name: str,
    job_slug: str,
    data_path: Path,
    model_dir: Path,
    metric_column: str,
    target_column: str,
    global_cfg: BaseGlobalConfig,
    epochs: int,
    seed: int,
    ensemble_size: int,
    train_enabled: bool,
    threshold_args: list[tuple[str, str]],
    classification_metrics: list[str],
    compound_min_screens: float | None,
    compound_screens_column: str,
    split_column: str,
    renderers: Renderers,
    screens_weight_mode: str,
) -> str:
    header = renderers.header(job_name, global_cfg)
    env_block = renderers.common_env(global_cfg, job_name)
    temp_parquet = f"${{TEMP_DIR}}/{job_slug}_threshold.parquet"
    extra_prep_lines: list[str] = []
    if threshold_args:
        for flag, value in threshold_args:
            extra_prep_lines.append(f"  {flag} {shlex.quote(value)}")
    if compound_min_screens is not None:
        min_value = format(compound_min_screens, "g")
        extra_prep_lines.append(f"  --compound-min-screens {min_value}")
    if compound_min_screens is not None or screens_weight_mode != "none":
        extra_prep_lines.append(f'  --compound-screens-column "{compound_screens_column}"')
    if screens_weight_mode != "none":
        extra_prep_lines.append(f'  --screens-weight-mode "{screens_weight_mode}"')

    lines = [
        header,
        "",
        env_block,
        "",
        f'SPLIT_COLUMN="{split_column}"',
        f'MODEL_DIR="{model_dir}"',
        'mkdir -p "${MODEL_DIR}"',
        f'DATA_PATH="{data_path}"',
        f'TEMP_PARQUET="{temp_parquet}"',
        "",
        'echo "[INFO] [$START] [$(date)] [$$] Preparing threshold-classifier dataset"',
    ]

    prep_lines = [
        '"${PYTHON_BIN}" -m model_jobs.cli prepare-threshold-classifier \\',
        '  --input "${DATA_PATH}" \\',
        '  --output "${TEMP_PARQUET}" \\',
        '  --smiles-column "smiles" \\',
        '  --split-column "${SPLIT_COLUMN}" \\',
        f'  --metric-column "{metric_column}" \\',
        f'  --target-column "{target_column}"',
    ]
    if extra_prep_lines:
        prep_lines[-1] += " \\"
        for i, line in enumerate(extra_prep_lines):
            if i < len(extra_prep_lines) - 1:
                prep_lines.append(f"{line} \\")
            else:
                prep_lines.append(line)
    lines.append("\n".join(prep_lines))

    if train_enabled:
        train_cmd_lines = [
            "${CHEMPROP_TRAIN} \\",
            '  --data-path "${TEMP_PARQUET}" \\',
            '  --save-dir "${MODEL_DIR}" \\',
            "  --task-type classification \\",
            f"  --ensemble-size {ensemble_size} \\",
            '  --smiles-columns "smiles" \\',
            f'  --target-columns "{target_column}" \\',
            '  --splits-column "${SPLIT_COLUMN}" \\',
        ]
        if screens_weight_mode != "none":
            train_cmd_lines.append(f'  --weight-column "{compound_screens_column}" \\')
        train_cmd_lines.extend(
            [
                f"  --epochs {epochs} \\",
                "  --show-individual-scores \\",
            ]
        )
        for metric in classification_metrics:
            train_cmd_lines.append(f'  --metrics "{metric}" \\')
        train_cmd_lines.append('  --auto-class-weights "balanced" \\')
        train_cmd_lines.append(f"  --pytorch-seed {seed} \\")
        train_cmd_lines.append(f"  --num-workers {global_cfg.cpus}")

        lines.extend(
            [
                "",
                'echo "[INFO] [$START] [$(date)] [$$] Starting Chemprop threshold-classifier training"',
                "",
                "\n".join(train_cmd_lines),
            ]
        )

    lines.extend(
        [
            "",
            'rm "${TEMP_PARQUET}" && echo "[INFO] [$START] [$(date)] [$$] Removed temporary file ${TEMP_PARQUET}"',
        ]
    )

    return "\n".join(lines).strip() + "\n"


# ---------------------------------------------------------------------------
# Public (generic) configuration
# ---------------------------------------------------------------------------


@dataclass
class GlobalConfig:
    models_dir: Path
    temp_dir: Path
    cpus: int
    conda_commands: list[str]
    conda_activate: str
    python_executable: str
    epochs: int
    seed: int
    metrics_classification: list[str]
    metrics_regression: list[str]
    chemprop_train_cmd: str
    submit: bool
    dataset_aliases: dict[str, Path] = field(default_factory=dict)
    base_path: Path = field(default_factory=Path)


def _parse_global_config(raw: dict, base_path: Path) -> GlobalConfig:
    if "models_dir" not in raw:
        raise ConfigError("global.models_dir is required.")
    if "temp_dir" not in raw:
        raise ConfigError("global.temp_dir is required.")

    dataset_aliases = {}
    for key, value in (raw.get("datasets") or {}).items():
        dataset_aliases[str(key)] = _resolve_path(value, base_path)

    return GlobalConfig(
        models_dir=_resolve_path(raw["models_dir"], base_path),
        temp_dir=_resolve_path(raw["temp_dir"], base_path),
        cpus=int(raw.get("cpus", raw.get("cores", 1))),
        conda_commands=[str(cmd) for cmd in raw.get("conda_commands", [])],
        conda_activate=str(raw.get("conda_activate", "")),
        python_executable=str(raw.get("python", "python")),
        epochs=int(raw.get("epochs", 50)),
        seed=int(raw.get("seed", 42)),
        metrics_classification=[
            str(m) for m in raw.get("classification_metrics", ["prc", "roc", "accuracy", "f1"])
        ],
        metrics_regression=[str(m) for m in raw.get("regression_metrics", ["rmse", "mae", "r2"])],
        chemprop_train_cmd=str(raw.get("chemprop_train_cmd", "chemprop train")),
        submit=bool(raw.get("submit", True)),
        dataset_aliases=dataset_aliases,
        base_path=base_path,
    )


def _render_header(job_name: str, global_cfg: GlobalConfig) -> str:
    # Generic header: only a POSIX shebang. UGE-specific directives live in
    # ``jobgen_uge.py`` for internal cluster use.
    return "#!/bin/bash"


def _common_env_base_lines(global_cfg: BaseGlobalConfig, job_name: str) -> list[str]:
    lines = [
        "set -euo pipefail",
        "",
        "START=$(date +%s)",
        "STARTDATE=$(date)",
        f'echo "[INFO] [$START] [$STARTDATE] [$$] Starting job {job_name}"',
        "",
        "cleanup() {",
        "  local exit_code=$?",
        "  local end=$(date +%s)",
        "  local end_date=$(date)",
        '  echo "[INFO] [$end] [$end_date] [$$] Job finished with code ${exit_code}"',
        '  echo "[INFO] [$end] [$end_date] [$$] Wall time (seconds): $(( end - START ))"',
        "}",
        "trap cleanup EXIT",
        "",
        f'TEMP_DIR="{global_cfg.temp_dir}"',
        'mkdir -p "${TEMP_DIR}"',
    ]
    return lines


def _append_conda_and_binaries(lines: list[str], global_cfg: BaseGlobalConfig) -> None:
    if global_cfg.conda_commands:
        lines.append("")
        lines.extend(global_cfg.conda_commands)
    if global_cfg.conda_activate:
        lines.append(global_cfg.conda_activate)

    lines.extend(
        [
            "",
            f'PYTHON_BIN="{global_cfg.python_executable}"',
            f'CHEMPROP_TRAIN="{global_cfg.chemprop_train_cmd}"',
            "",
        ]
    )


def _render_common_env(global_cfg: GlobalConfig, job_name: str) -> str:
    lines = _common_env_base_lines(global_cfg, job_name)
    _append_conda_and_binaries(lines, global_cfg)
    return "\n".join(lines)


_PUBLIC_RENDERERS = Renderers(
    header=_render_header,
    common_env=_render_common_env,
)


def load_submission_plan(config_path: Path) -> tuple[GlobalConfig, list[JobDefinition]]:
    return _load_submission_plan(
        config_path,
        parse_global_config=_parse_global_config,
        renderers=_PUBLIC_RENDERERS,
    )
