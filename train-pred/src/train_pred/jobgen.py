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
    chemprop_predict_cmd: str
    submit: bool
    dataset_aliases: dict[str, Path]
    base_path: Path


GC = TypeVar("GC", bound=BaseGlobalConfig)


@dataclass
class Renderers:
    header: Callable[[str, GC], str]
    common_env: Callable[[GC, str], str]


@dataclass
class PredictionSet:
    name: str
    input_path: Path
    preds_path: Path
    filters: list[str] = field(default_factory=list)
    compound_min_screens: float | None = None
    compound_screens_column: str = "screens"

    @property
    def slug(self) -> str:
        return _slugify(self.name)


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
            job_defs = _build_prediction_jobs(entry, global_cfg, renderers)
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


def _resolve_dataset_path(value: str | Path | None, global_cfg: BaseGlobalConfig, field_name: str) -> Path:
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


def _format_model_dir(raw_value: str | None, global_cfg: BaseGlobalConfig, context: dict[str, str]) -> Path:
    if raw_value is None:
        return global_cfg.models_dir / context["job_slug"]
    formatted = raw_value.format(**context)
    return _resolve_path(formatted, global_cfg.base_path)


def _collect_prediction_sets(
    task: dict,
    model_dir: Path,
    global_cfg: BaseGlobalConfig,
    default_sets: list[tuple[str, str | Path]] | None = None,
) -> list[PredictionSet]:
    prediction_sets: list[PredictionSet] = []
    default_min = task.get("prediction_compound_min_screens")
    default_screens_column = task.get(
        "prediction_compound_screens_column",
        task.get("compound_screens_column", "screens"),
    )
    if default_screens_column is None:
        default_screens_column = "screens"
    default_screens_column = str(default_screens_column)

    def _coerce_min(value: object | None) -> float | None:
        if value is None:
            return None
        return float(value)

    default_min_value = _coerce_min(default_min)

    if default_sets:
        for name, path_value in default_sets:
            dataset_path = _resolve_dataset_path(path_value, global_cfg, f"{name}_path")
            preds_path = model_dir / "predictions" / f"{_slugify(name)}_preds.csv"
            prediction_sets.append(
                PredictionSet(
                    name=name,
                    input_path=dataset_path,
                    preds_path=preds_path,
                    filters=[],
                    compound_min_screens=default_min_value,
                    compound_screens_column=default_screens_column,
                )
            )

    for spec in task.get("prediction_sets", []):
        if not isinstance(spec, dict):
            raise ConfigError("prediction_sets entries must be mappings.")
        name = spec.get("name")
        if not name:
            raise ConfigError("prediction_sets entry missing 'name'.")
        dataset_path = _resolve_dataset_path(spec.get("input_path"), global_cfg, f"prediction_sets[{name}].input_path")
        preds_value = spec.get("output_path")
        if preds_value:
            preds_path = _resolve_path(preds_value, global_cfg.base_path)
        else:
            preds_path = model_dir / "predictions" / f"{_slugify(name)}_preds.csv"
        filters_raw = spec.get("filters", [])
        if isinstance(filters_raw, dict):
            filters_list = [f"{key}={value}" for key, value in filters_raw.items()]
        else:
            filters_list = [str(v) for v in filters_raw]
        set_min = spec.get("compound_min_screens")
        set_screens_column = spec.get("compound_screens_column", default_screens_column)
        if set_screens_column is None:
            set_screens_column = default_screens_column
        prediction_sets.append(
            PredictionSet(
                name=str(name),
                input_path=dataset_path,
                preds_path=preds_path,
                filters=filters_list,
                compound_min_screens=_coerce_min(set_min) if set_min is not None else default_min_value,
                compound_screens_column=str(set_screens_column),
            )
        )

    return prediction_sets


# ---------------------------------------------------------------------------
# Job builders
# ---------------------------------------------------------------------------


def _build_multilabel_jobs(task: dict, global_cfg: GC, renderers: Renderers) -> list[JobDefinition]:
    job_name = _extract_job_name(task)
    job_slug = _slugify(job_name)

    trainval_path = _resolve_dataset_path(task.get("trainval_path"), global_cfg, "trainval_path")
    keep_columns = ["smiles", "split"]
    keep_columns.extend(task.get("extra_keep_columns", []))
    keep_columns = list(dict.fromkeys(map(str, keep_columns)))
    epochs = int(task.get("epochs", global_cfg.epochs))
    seed = int(task.get("seed", global_cfg.seed))
    ensemble_size = int(task.get("ensemble_size", 1))
    predict_cal = bool(task.get("predict_calibration", True))
    predict_test = bool(task.get("predict_test", False))

    model_context = {
        "job_name": job_name,
        "job_slug": job_slug,
    }
    model_dir = _format_model_dir(task.get("model_dir"), global_cfg, model_context)

    default_prediction_sets: list[tuple[str, str | Path]] = []
    if predict_cal:
        default_prediction_sets.append(("calibration", task.get("calibration_path")))
    if predict_test:
        default_prediction_sets.append(("test", task.get("test_path")))

    prediction_sets = _collect_prediction_sets(task, model_dir, global_cfg, default_prediction_sets)

    submit_flag = _should_submit(task, global_cfg)
    content = _render_multilabel_script(
        job_name=job_name,
        job_slug=job_slug,
        trainval_path=trainval_path,
        model_dir=model_dir,
        keep_columns=keep_columns,
        prediction_sets=prediction_sets,
        global_cfg=global_cfg,
        epochs=epochs,
        seed=seed,
        ensemble_size=ensemble_size,
        train_enabled=bool(task.get("train", True)),
        classification_metrics=_task_metrics(task, "classification_metrics", global_cfg.metrics_classification),
        renderers=renderers,
    )
    filename = f"{job_slug}.sh"
    return [JobDefinition(job_name=job_name, filename=filename, content=content, submit=submit_flag)]


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
            raise ConfigError("Regression task requires 'target_column' or 'targets'.")
        targets_list = [str(target)]

    epochs = int(task.get("epochs", global_cfg.epochs))
    seed = int(task.get("seed", global_cfg.seed))
    ensemble_size = int(task.get("ensemble_size", 1))
    predict_cal = bool(task.get("predict_calibration", True))
    predict_test = bool(task.get("predict_test", False))
    compound_min = task.get("compound_min_screens")
    compound_min_value = float(compound_min) if compound_min is not None else None
    compound_screens_column = task.get("compound_screens_column", "screens")
    if compound_screens_column is None:
        compound_screens_column = "screens"
    compound_screens_column = str(compound_screens_column)

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

        default_prediction_sets: list[tuple[str, str | Path]] = []
        if predict_cal:
            default_prediction_sets.append(("calibration", task.get("calibration_path")))
        if predict_test:
            default_prediction_sets.append(("test", task.get("test_path")))

        prediction_sets = _collect_prediction_sets(task, model_dir, global_cfg, default_prediction_sets)

        submit_flag = _should_submit(task, global_cfg)
        content = _render_regression_script(
            job_name=per_job_name,
            job_slug=per_slug,
            trainval_path=trainval_path,
            model_dir=model_dir,
            target_column=target,
            task_type=task_type,
            prediction_sets=prediction_sets,
            global_cfg=global_cfg,
            epochs=epochs,
            seed=seed,
            ensemble_size=ensemble_size,
            train_enabled=bool(task.get("train", True)),
            regression_metrics=_regression_metrics(task, global_cfg),
            compound_min_screens=compound_min_value,
            compound_screens_column=compound_screens_column,
            renderers=renderers,
        )
        filename = f"{per_slug}.sh"
        job_defs.append(JobDefinition(per_job_name, filename, content, submit_flag))

    return job_defs


def _build_threshold_jobs(task: dict, global_cfg: GC, renderers: Renderers) -> list[JobDefinition]:
    job_name = _extract_job_name(task)
    job_slug = _slugify(job_name)
    trainval_path = _resolve_dataset_path(task.get("trainval_path"), global_cfg, "trainval_path")
    metric_column = str(task.get("metric_column", "score"))
    target_column = str(task.get("target_column", "target"))
    predict_cal = bool(task.get("predict_calibration", True))
    predict_test = bool(task.get("predict_test", False))
    epochs = int(task.get("epochs", global_cfg.epochs))
    seed = int(task.get("seed", global_cfg.seed))
    ensemble_size = int(task.get("ensemble_size", 1))
    compound_min = task.get("compound_min_screens")
    compound_min_value = float(compound_min) if compound_min is not None else None
    compound_screens_column = task.get("compound_screens_column", "screens")
    if compound_screens_column is None:
        compound_screens_column = "screens"
    compound_screens_column = str(compound_screens_column)

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
            raise ConfigError("Threshold specification requires lower_percentile or lower_threshold.")
        if upper_percentile is None and upper_threshold is None:
            raise ConfigError("Threshold specification requires upper_percentile or upper_threshold.")

        suffix = spec.get("suffix") or spec.get("name_suffix")
        if suffix:
            suffix_str = str(suffix)
        elif lower_percentile is not None and upper_percentile is not None:
            suffix_str = f"p{lower_percentile}_{upper_percentile}"
        elif lower_threshold is not None and upper_threshold is not None:
            suffix_str = f"thr{lower_threshold}_{upper_threshold}"
        else:
            suffix_str = f"combo{idx+1}"

        per_job_name = f"{job_name}_{suffix_str}"
        per_slug = _slugify(per_job_name)
        context = {
            "job_name": per_job_name,
            "job_slug": per_slug,
            "suffix": suffix_str,
            "metric": metric_column,
        }
        model_dir = _format_model_dir(task.get("model_dir"), global_cfg, context)

        default_prediction_sets: list[tuple[str, str | Path]] = []
        if predict_cal:
            default_prediction_sets.append(("calibration", task.get("calibration_path")))
        if predict_test:
            default_prediction_sets.append(("test", task.get("test_path")))
        prediction_sets = _collect_prediction_sets(task, model_dir, global_cfg, default_prediction_sets)

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
            trainval_path=trainval_path,
            model_dir=model_dir,
            metric_column=metric_column,
            target_column=target_column,
            prediction_sets=prediction_sets,
            global_cfg=global_cfg,
            epochs=epochs,
            seed=seed,
            ensemble_size=ensemble_size,
            train_enabled=bool(task.get("train", True)),
            threshold_args=threshold_args,
            classification_metrics=_task_metrics(task, "classification_metrics", global_cfg.metrics_classification),
            compound_min_screens=compound_min_value,
            compound_screens_column=compound_screens_column,
            renderers=renderers,
        )
        filename = f"{per_slug}.sh"
        jobs.append(JobDefinition(per_job_name, filename, content, submit_flag))

    return jobs


def _build_prediction_jobs(task: dict, global_cfg: GC, renderers: Renderers) -> list[JobDefinition]:
    job_name = _extract_job_name(task)
    job_slug = _slugify(job_name)
    model_context = {"job_name": job_name, "job_slug": job_slug}
    model_dir = _format_model_dir(task.get("model_dir"), global_cfg, model_context)
    model_path_value = task.get("model_path")
    model_path = _resolve_dataset_path(model_path_value, global_cfg, "model_path") if model_path_value else model_dir

    prediction_sets = task.get("prediction_sets")
    if not prediction_sets:
        raise ConfigError("Prediction-only task must provide at least one prediction_sets entry.")

    prediction_set_objs = _collect_prediction_sets({"prediction_sets": prediction_sets}, model_dir, global_cfg)
    submit_flag = _should_submit(task, global_cfg)
    content = _render_prediction_only_script(
        job_name=job_name,
        job_slug=job_slug,
        model_path=model_path,
        prediction_sets=prediction_set_objs,
        global_cfg=global_cfg,
        renderers=renderers,
    )
    filename = f"{job_slug}.sh"
    return [JobDefinition(job_name=job_name, filename=filename, content=content, submit=submit_flag)]


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


# ---------------------------------------------------------------------------
# Script rendering
# ---------------------------------------------------------------------------


def _render_prediction_blocks(prediction_sets: list[PredictionSet], global_cfg: BaseGlobalConfig, job_slug: str) -> str:
    blocks: list[str] = []
    for pred_set in prediction_sets:
        temp_csv = f'${{TEMP_DIR}}/{job_slug}_{pred_set.slug}.csv'
        preds_path = pred_set.preds_path
        preds_dir_cmd = f"mkdir -p $(dirname {shlex.quote(str(preds_path))})"
        cmd_lines = [
            f'"${{PYTHON_BIN}}" -m train_pred.cli prepare-smiles-csv \\',
            f'  --input "{pred_set.input_path}" \\',
            f'  --output "{temp_csv}" \\',
            '  --smiles-column "smiles"',
        ]
        for flt in pred_set.filters:
            cmd_lines[-1] += " \\"
            cmd_lines.append(f"  --filter '{flt}'")
        if pred_set.compound_min_screens is not None:
            min_value = format(pred_set.compound_min_screens, "g")
            cmd_lines[-1] += " \\"
            cmd_lines.append(f"  --compound-min-screens {min_value} \\")
            cmd_lines.append(
                f'  --compound-screens-column "{pred_set.compound_screens_column}"'
            )
        prepare_cmd = "\n".join(cmd_lines)

        predict_lines = [
            f'${{CHEMPROP_PREDICT}} \\',
            f'  --test-path "{temp_csv}" \\',
            f'  --preds-path "{preds_path}" \\',
            '  --model-path "${MODEL_PATH}" \\',
            '  --drop-extra-columns \\',
            '  --smiles-columns "smiles" \\',
            f"  --num-workers {global_cfg.cpus}",
        ]

        block_lines = [
            f'echo "[INFO] [$START] [$(date)] [$$] Preparing dataset \'{pred_set.name}\' for prediction"',
            "",
            prepare_cmd,
            "",
            preds_dir_cmd,
            "",
            f'echo "[INFO] [$START] [$(date)] [$$] Running Chemprop prediction for \'{pred_set.name}\'"',
            "",
            "\n".join(predict_lines),
            "",
            f'rm "{temp_csv}" && echo "[INFO] [$START] [$(date)] [$$] Removed temporary file {temp_csv}"',
        ]
        block = "\n".join(block_lines).strip()
        blocks.append(block)
    return "\n\n".join(blocks)


def _render_multilabel_script(
    *,
    job_name: str,
    job_slug: str,
    trainval_path: Path,
    model_dir: Path,
    keep_columns: list[str],
    prediction_sets: list[PredictionSet],
    global_cfg: BaseGlobalConfig,
    epochs: int,
    seed: int,
    ensemble_size: int,
    train_enabled: bool,
    classification_metrics: list[str],
    renderers: Renderers,
) -> str:
    header = renderers.header(job_name, global_cfg)
    env_block = renderers.common_env(global_cfg, job_name)

    keep_args = "\n".join(
        f'          --keep-column "{col}"' for col in keep_columns if col not in {"smiles", "split"}
    )
    temp_parquet = f'${{TEMP_DIR}}/{job_slug}_multilabel_trimmed.parquet'

    body_lines: list[str] = [header, "", env_block, "", f'MODEL_DIR="{model_dir}"', 'mkdir -p "${MODEL_DIR}"']
    body_lines.append(f'TRAINVAL_PATH="{trainval_path}"')
    body_lines.append(f'TEMP_PARQUET="{temp_parquet}"')
    body_lines.append("")
    body_lines.append('echo "[INFO] [$START] [$(date)] [$$] Trimming multi-task dataset"')

    trim_command = textwrap.dedent(
        f"""
        "${{PYTHON_BIN}}" -m train_pred.cli trim-columns \\
          --input "${{TRAINVAL_PATH}}" \\
          --output "${{TEMP_PARQUET}}" \\
          --keep-column "smiles" \\
          --keep-column "split" \\
          --keep-numeric-columns
{keep_args}
        """
    ).strip()
    body_lines.append(trim_command)

    if train_enabled:
        train_cmd_lines = [
            "${CHEMPROP_TRAIN} \\",
            '  --data-path "${TEMP_PARQUET}" \\',
            '  --save-dir "${MODEL_DIR}" \\',
            '  --task-type classification \\',
            f"  --ensemble-size {ensemble_size} \\",
            '  --smiles-columns "smiles" \\',
            '  --splits-column "split" \\',
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

    if prediction_sets:
        body_lines.extend(
            [
                "",
                'MODEL_PATH="${MODEL_DIR}"',
                "",
                _render_prediction_blocks(prediction_sets, global_cfg, job_slug),
            ]
        )

    return "\n".join(body_lines).strip() + "\n"


def _render_regression_script(
    *,
    job_name: str,
    job_slug: str,
    trainval_path: Path,
    model_dir: Path,
    target_column: str,
    task_type: str,
    prediction_sets: list[PredictionSet],
    global_cfg: BaseGlobalConfig,
    epochs: int,
    seed: int,
    ensemble_size: int,
    train_enabled: bool,
    regression_metrics: list[str],
    compound_min_screens: float | None,
    compound_screens_column: str,
    renderers: Renderers,
) -> str:
    header = renderers.header(job_name, global_cfg)
    env_block = renderers.common_env(global_cfg, job_name)
    temp_parquet = f'${{TEMP_DIR}}/{job_slug}_regression.parquet'
    extra_prep_lines: list[str] = []
    if compound_min_screens is not None:
        min_value = format(compound_min_screens, "g")
        extra_prep_lines.append(f"  --compound-min-screens {min_value}")
        extra_prep_lines.append(f'  --compound-screens-column "{compound_screens_column}"')

    lines = [
        header,
        "",
        env_block,
        "",
        f'MODEL_DIR="{model_dir}"',
        'mkdir -p "${MODEL_DIR}"',
        f'TRAINVAL_PATH="{trainval_path}"',
        f'TEMP_PARQUET="{temp_parquet}"',
        "",
    ]

    lines.append('echo "[INFO] [$START] [$(date)] [$$] Preparing regression dataset"')
    prep_lines = [
        '"${PYTHON_BIN}" -m train_pred.cli prepare-regression \\',
        '  --input "${TRAINVAL_PATH}" \\',
        '  --output "${TEMP_PARQUET}" \\',
        '  --smiles-column "smiles" \\',
        '  --split-column "split" \\',
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
            '  --splits-column "split" \\',
            f'  --weight-column "{compound_screens_column}" \\',
            f"  --epochs {epochs} \\",
            "  --show-individual-scores \\",
        ]
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

    if prediction_sets:
        lines.extend(
            [
                "",
                'MODEL_PATH="${MODEL_DIR}"',
                "",
                _render_prediction_blocks(prediction_sets, global_cfg, job_slug),
            ]
        )

    return "\n".join(lines).strip() + "\n"


def _render_threshold_script(
    *,
    job_name: str,
    job_slug: str,
    trainval_path: Path,
    model_dir: Path,
    metric_column: str,
    target_column: str,
    prediction_sets: list[PredictionSet],
    global_cfg: BaseGlobalConfig,
    epochs: int,
    seed: int,
    ensemble_size: int,
    train_enabled: bool,
    threshold_args: list[tuple[str, str]],
    classification_metrics: list[str],
    compound_min_screens: float | None,
    compound_screens_column: str,
    renderers: Renderers,
) -> str:
    header = renderers.header(job_name, global_cfg)
    env_block = renderers.common_env(global_cfg, job_name)
    temp_parquet = f'${{TEMP_DIR}}/{job_slug}_threshold.parquet'
    extra_prep_lines: list[str] = []
    if threshold_args:
        for flag, value in threshold_args:
            extra_prep_lines.append(f"  {flag} {shlex.quote(value)}")
    if compound_min_screens is not None:
        min_value = format(compound_min_screens, "g")
        extra_prep_lines.append(f"  --compound-min-screens {min_value}")
        extra_prep_lines.append(f'  --compound-screens-column "{compound_screens_column}"')

    lines = [
        header,
        "",
        env_block,
        "",
        f'MODEL_DIR="{model_dir}"',
        'mkdir -p "${MODEL_DIR}"',
        f'TRAINVAL_PATH="{trainval_path}"',
        f'TEMP_PARQUET="{temp_parquet}"',
        "",
        'echo "[INFO] [$START] [$(date)] [$$] Preparing threshold-classifier dataset"',
    ]

    prep_lines = [
        '"${PYTHON_BIN}" -m train_pred.cli prepare-threshold-classifier \\',
        '  --input "${TRAINVAL_PATH}" \\',
        '  --output "${TEMP_PARQUET}" \\',
        '  --smiles-column "smiles" \\',
        '  --split-column "split" \\',
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
            '  --splits-column "split" \\',
            f'  --weight-column "{compound_screens_column}" \\',
            f"  --epochs {epochs} \\",
            "  --show-individual-scores \\",
        ]
        for metric in classification_metrics:
            train_cmd_lines.append(f'  --metrics "{metric}" \\')
        train_cmd_lines.append("  --auto-class-weights \"balanced\" \\")
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

    if prediction_sets:
        lines.extend(
            [
                "",
                'MODEL_PATH="${MODEL_DIR}"',
                "",
                _render_prediction_blocks(prediction_sets, global_cfg, job_slug),
            ]
        )

    return "\n".join(lines).strip() + "\n"


def _render_prediction_only_script(
    *,
    job_name: str,
    job_slug: str,
    model_path: Path,
    prediction_sets: list[PredictionSet],
    global_cfg: BaseGlobalConfig,
    renderers: Renderers,
) -> str:
    if not prediction_sets:
        raise ConfigError("Prediction-only job requires at least one prediction set.")
    header = renderers.header(job_name, global_cfg)
    env_block = renderers.common_env(global_cfg, job_name)
    lines = [
        header,
        "",
        env_block,
        "",
        f'MODEL_PATH="{model_path}"',
        "",
        _render_prediction_blocks(prediction_sets, global_cfg, job_slug),
    ]
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
    chemprop_predict_cmd: str
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
        metrics_classification=[str(m) for m in raw.get(
            "classification_metrics", ["prc", "roc", "accuracy", "f1"]
        )],
        metrics_regression=[str(m) for m in raw.get(
            "regression_metrics", ["rmse", "mae", "r2"]
        )],
        chemprop_train_cmd=str(raw.get("chemprop_train_cmd", "chemprop train")),
        chemprop_predict_cmd=str(raw.get("chemprop_predict_cmd", "chemprop predict")),
        submit=bool(raw.get("submit", True)),
        dataset_aliases=dataset_aliases,
        base_path=base_path,
    )


def _render_header(job_name: str, global_cfg: GlobalConfig) -> str:
    # Generic header: only a POSIX shebang. UGE-specific directives live in
    # ``jobgen_uge.py`` for internal cluster use.
    return "#!/bin/bash"


def _render_common_env(global_cfg: GlobalConfig, job_name: str) -> str:
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
            f'CHEMPROP_PREDICT="{global_cfg.chemprop_predict_cmd}"',
            "",
        ]
    )
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
