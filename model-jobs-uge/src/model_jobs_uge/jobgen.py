from __future__ import annotations

import copy
import re
import shlex
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence, TypeVar

import yaml

__all__ = ["ConfigError", "GlobalConfig", "JobDefinition", "load_submission_plan"]


class ConfigError(Exception):
    """Raised when the submission configuration file is invalid."""


class BaseGlobalConfig(Protocol):
    email: str
    logs_dir: Path
    models_dir: Path
    temp_dir: Path
    cpus: int
    memory: str
    runtime: str
    gpu_cards: int
    queue_directives: list[str]
    conda_commands: list[str]
    conda_activate: str
    python_executable: str
    epochs: int
    notify_events: str
    email_notifications: bool
    gpu_map_script: str | None
    metrics_classification: list[str]
    metrics_regression: list[str]
    chemprop_train_cmd: str
    submit: bool
    dataset_aliases: dict[str, Path]
    extra_header_directives: list[str]
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


@dataclass
class GlobalConfig:
    email: str
    logs_dir: Path
    models_dir: Path
    temp_dir: Path
    cpus: int
    memory: str
    runtime: str
    gpu_cards: int
    queue_directives: list[str]
    conda_commands: list[str]
    conda_activate: str
    python_executable: str
    epochs: int
    notify_events: str
    email_notifications: bool
    gpu_map_script: str | None
    metrics_classification: list[str]
    metrics_regression: list[str]
    chemprop_train_cmd: str
    submit: bool
    dataset_aliases: dict[str, Path] = field(default_factory=dict)
    extra_header_directives: list[str] = field(default_factory=list)
    base_path: Path = field(default_factory=Path)


def load_submission_plan(config_path: Path) -> tuple[GlobalConfig, list[JobDefinition]]:
    raw = _load_yaml(config_path)
    base_path = config_path.resolve().parent
    sweeps = _parse_sweeps(raw.get("sweeps") or {})
    defaults = _parse_defaults(raw.get("defaults") or {})
    global_cfg = _parse_global_config(raw.get("global") or {}, base_path)

    tasks_raw = raw.get("tasks")
    if not isinstance(tasks_raw, list) or not tasks_raw:
        raise ConfigError("The configuration must include a non-empty 'tasks' list.")

    jobs: list[JobDefinition] = []
    for entry in tasks_raw:
        for task in _expand_task_entry(entry, sweeps, defaults, global_cfg):
            task_type = _normalize_task_type(task.get("type"))
            if task_type == "multilabel":
                jobs.extend(_build_multilabel_jobs(task, global_cfg, _UGE_RENDERERS))
            elif task_type == "regression":
                jobs.extend(_build_regression_jobs(task, global_cfg, _UGE_RENDERERS))
            elif task_type == "threshold":
                jobs.extend(_build_threshold_jobs(task, global_cfg, _UGE_RENDERERS))
            else:
                raise ConfigError(f"Unknown task type '{task.get('type')}'.")

    return global_cfg, jobs


# ---------------------------------------------------------------------------
# Config loading and expansion
# ---------------------------------------------------------------------------


def _load_yaml(config_path: Path) -> dict:
    try:
        raw = yaml.safe_load(config_path.read_text())
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration file not found: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Unable to parse YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError("Configuration must be a mapping.")
    return raw


def _parse_sweeps(raw: Mapping[str, object]) -> dict[str, list[object]]:
    if not isinstance(raw, Mapping):
        raise ConfigError("sweeps must be a mapping if provided.")
    sweeps: dict[str, list[object]] = {}
    for key, values in raw.items():
        if isinstance(values, list):
            sweeps[str(key)] = list(values)
            continue
        sweeps[str(key)] = [values]
    return sweeps


def _parse_defaults(raw: Mapping[str, object]) -> dict[str, dict]:
    defaults: dict[str, dict] = {
        "all": {},
        "multilabel": {},
        "regression": {},
        "threshold": {},
        "hit_rate": {},
    }
    if not raw:
        return defaults
    if not isinstance(raw, Mapping):
        raise ConfigError("defaults must be a mapping if provided.")
    for key in defaults:
        section = raw.get(key, {})
        if section is None:
            continue
        if not isinstance(section, Mapping):
            raise ConfigError(f"defaults.{key} must be a mapping.")
        defaults[key] = copy.deepcopy(dict(section))
    return defaults


def _apply_task_defaults(task_entry: Mapping[str, object], defaults: dict[str, dict]) -> dict:
    task_type = _normalize_task_type(task_entry.get("type"))
    merged: dict = copy.deepcopy(defaults.get("all", {}))
    merged.update(copy.deepcopy(defaults.get(task_type, {})))
    if task_type in {"regression", "threshold"}:
        merged.update(copy.deepcopy(defaults.get("hit_rate", {})))
    merged.update(task_entry)
    return merged


def _normalize_task_type(value: object) -> str:
    if value is None:
        raise ConfigError("Task is missing required 'type' field.")
    normalized = str(value).replace("-", "_").lower()
    if normalized in {"multi_label", "multilabel"}:
        return "multilabel"
    if normalized in {"regression", "threshold"}:
        return normalized
    raise ConfigError(f"Unknown task type '{value}'.")


def _expand_values(raw_value: object, sweeps: Mapping[str, list[object]]) -> list[object]:
    if isinstance(raw_value, str) and raw_value.startswith("@"):
        sweep_key = raw_value[1:]
        if sweep_key not in sweeps:
            raise ConfigError(f"Unknown sweep '{sweep_key}' referenced in expand.")
        return list(sweeps[sweep_key])
    if isinstance(raw_value, list):
        return list(raw_value)
    return [raw_value]


def _expand_matrix(expand_spec: object, sweeps: Mapping[str, list[object]]) -> list[dict[str, object]]:
    if expand_spec is None:
        return [{}]
    if not isinstance(expand_spec, Mapping):
        raise ConfigError("expand must be a mapping of variable -> values.")

    keys: list[str] = []
    values: list[list[object]] = []
    for key, raw in expand_spec.items():
        keys.append(str(key))
        values.append(_expand_values(raw, sweeps))

    expanded: list[dict[str, object]] = []
    for combo in product(*values):
        expanded.append(dict(zip(keys, combo)))
    return expanded or [{}]


def _format_strings(value: object, context: Mapping[str, object]) -> object:
    if isinstance(value, str):
        try:
            return value.format(**context)
        except KeyError as exc:
            raise ConfigError(f"Missing variable {exc} in formatted value '{value}'.") from exc
    if isinstance(value, list):
        return [_format_strings(item, context) for item in value]
    if isinstance(value, Mapping):
        return {k: _format_strings(v, context) for k, v in value.items()}
    return value


def _expand_thresholds(
    raw_thresholds: object,
    sweeps: Mapping[str, list[object]],
    context: Mapping[str, object],
) -> list[dict]:
    if raw_thresholds is None:
        raise ConfigError("Threshold tasks require a 'thresholds' block.")

    if isinstance(raw_thresholds, Mapping) and "template" in raw_thresholds:
        expand_spec = raw_thresholds.get("expand", {})
        template = raw_thresholds.get("template", {})
        if not isinstance(template, Mapping):
            raise ConfigError("thresholds.template must be a mapping.")
        entries: list[dict] = []
        for combo in _expand_matrix(expand_spec, sweeps):
            merged_ctx = {**context, **combo}
            entries.append(_format_strings(template, merged_ctx))
        return entries

    if not isinstance(raw_thresholds, list) or not raw_thresholds:
        raise ConfigError("thresholds must be a non-empty list or an object with template/expand.")

    entries: list[dict] = []
    for spec in raw_thresholds:
        if not isinstance(spec, Mapping):
            raise ConfigError("Each thresholds entry must be a mapping.")
        entry_expand = spec.get("expand")
        base_spec = {k: v for k, v in spec.items() if k != "expand"}
        combos = _expand_matrix(entry_expand, sweeps)
        if combos:
            for combo in combos:
                merged_ctx = {**context, **combo}
                entries.append(_format_strings(base_spec, merged_ctx))
        else:
            entries.append(_format_strings(base_spec, context))
    return entries


def _expand_task_entry(
    task_entry: object,
    sweeps: Mapping[str, list[object]],
    defaults: dict[str, dict],
    global_cfg: GlobalConfig,
) -> list[dict]:
    if not isinstance(task_entry, Mapping):
        raise ConfigError("Each task entry must be a mapping.")

    merged = _apply_task_defaults(task_entry, defaults)
    expand_spec = merged.pop("expand", None)
    task_type = _normalize_task_type(merged.get("type"))

    expanded_tasks: list[dict] = []
    for combo in _expand_matrix(expand_spec, sweeps):
        ctx = dict(combo)
        thresholds_raw = merged.get("thresholds")
        task_body = {k: v for k, v in merged.items() if k != "thresholds"}

        formatted = _format_strings(task_body, ctx)
        formatted["type"] = task_type

        if task_type == "threshold":
            thresholds = _expand_thresholds(
                thresholds_raw if thresholds_raw is not None else defaults.get("threshold", {}).get("thresholds"),
                sweeps,
                ctx,
            )
            formatted["thresholds"] = thresholds
        elif thresholds_raw is not None:
            formatted["thresholds"] = _format_strings(thresholds_raw, ctx)

        if "split_seed" not in formatted:
            raise ConfigError("Each task must provide split_seed (or set it via defaults/expand).")

        expanded_tasks.append(formatted)

    return expanded_tasks


def _parse_global_config(raw: Mapping[str, object], base_path: Path) -> GlobalConfig:
    required = ["email", "logs_dir", "models_dir", "temp_dir", "memory", "runtime"]
    missing = [k for k in required if k not in raw]
    if missing:
        raise ConfigError(f"Missing required global fields: {', '.join(missing)}.")

    cpus_raw = raw.get("cpus", raw.get("cores"))
    if cpus_raw is None:
        raise ConfigError("global.cpus is required for UGE submissions.")
    gpu_raw = raw.get("gpu_cards", raw.get("gpu"))
    if gpu_raw is None:
        raise ConfigError("global.gpu_cards (or gpu) is required for UGE submissions.")

    dataset_aliases: dict[str, Path] = {}
    for key, value in (raw.get("datasets") or {}).items():
        dataset_aliases[str(key)] = _resolve_path(value, base_path)

    queue_directives = raw.get("queue_directives") or ["-j y", "-R y", "-notify"]

    return GlobalConfig(
        email=str(raw["email"]),
        logs_dir=_resolve_path(raw["logs_dir"], base_path),
        models_dir=_resolve_path(raw["models_dir"], base_path),
        temp_dir=_resolve_path(raw["temp_dir"], base_path),
        cpus=int(cpus_raw),
        memory=str(raw["memory"]),
        runtime=str(raw["runtime"]),
        gpu_cards=int(gpu_raw),
        queue_directives=[str(d) for d in queue_directives],
        conda_commands=[str(cmd) for cmd in raw.get("conda_commands", [])],
        conda_activate=str(raw.get("conda_activate", "")),
        python_executable=str(raw.get("python", "python")),
        epochs=int(raw.get("epochs", 50)),
        notify_events=str(raw.get("notify_events", "bea")),
        email_notifications=bool(raw.get("email_notifications", True)),
        gpu_map_script=str(raw.get("gpu_map_script")) if raw.get("gpu_map_script") else None,
        metrics_classification=[
            str(m) for m in raw.get("classification_metrics", ["prc", "roc", "accuracy", "f1"])
        ],
        metrics_regression=[str(m) for m in raw.get("regression_metrics", ["rmse", "mae", "r2"])],
        chemprop_train_cmd=str(raw.get("chemprop_train_cmd", "chemprop train")),
        submit=bool(raw.get("submit", True)),
        dataset_aliases=dataset_aliases,
        extra_header_directives=[str(d) for d in raw.get("extra_directives", [])],
        base_path=base_path,
    )


# ---------------------------------------------------------------------------
# Shared task helpers
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


def _task_data_path(task: Mapping[str, object], global_cfg: BaseGlobalConfig) -> Path:
    raw_path = task.get("data_path", task.get("trainval_path"))
    if raw_path is None:
        raise ConfigError("Task is missing required 'data_path'.")
    return _resolve_dataset_path(raw_path, global_cfg, "data_path")


def _extract_job_name(task: Mapping[str, object]) -> str:
    name = task.get("job_name") or task.get("name")
    if not name:
        raise ConfigError("Task is missing 'job_name' (or 'name').")
    return str(name)


def _slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip())
    return slug.strip("_") or "job"


def _format_model_dir(
    raw_value: str | None, global_cfg: BaseGlobalConfig, context: dict[str, str]
) -> Path:
    if raw_value is None:
        return global_cfg.models_dir / context["job_slug"]
    formatted = raw_value.format(**context)
    return _resolve_path(formatted, global_cfg.base_path)


def _split_column(split_seed: int) -> str:
    return f"split_seed{split_seed}"


def _ensure_seed_suffix(value: str, split_seed: int, *, delimiter: str = "_seed") -> str:
    suffix = f"{delimiter}{split_seed}"
    if value.endswith(suffix):
        return value
    return f"{value}{suffix}"


def _screens_weight_mode_from_task(task: Mapping[str, object]) -> str:
    mode = str(task.get("screens_weight_mode", "none")).strip().lower()
    if mode not in {"none", "linear", "sqrt"}:
        raise ConfigError("screens_weight_mode must be one of: none, linear, sqrt.")
    return mode


def _should_submit(task: Mapping[str, object], global_cfg: BaseGlobalConfig) -> bool:
    submit_override = task.get("submit")
    if submit_override is None:
        return global_cfg.submit
    return bool(submit_override)


# ---------------------------------------------------------------------------
# Job builders
# ---------------------------------------------------------------------------


def _build_multilabel_jobs(task: Mapping[str, object], global_cfg: GC, renderers: Renderers) -> list[JobDefinition]:
    _reject_overrides(task, ["split_column", "score_column", "score_columns"])
    job_name_base = _extract_job_name(task)
    data_path = _task_data_path(task, global_cfg)
    epochs = int(task.get("epochs", global_cfg.epochs))
    split_seed = int(task["split_seed"])
    chemprop_seed = int(task.get("chemprop_seed", split_seed))
    job_name_seeded = _ensure_seed_suffix(job_name_base, split_seed)
    job_slug = _slugify(job_name_seeded)
    split_column = _split_column(split_seed)
    keep_columns = ["smiles", split_column]
    keep_columns.extend(task.get("extra_keep_columns", []))
    keep_columns = list(dict.fromkeys(map(str, keep_columns)))
    ensemble_size = int(task.get("ensemble_size", 1))

    model_context = {
        "job_name": job_name_seeded,
        "job_slug": job_slug,
    }
    model_dir = _format_model_dir(task.get("model_dir"), global_cfg, model_context)

    submit_flag = _should_submit(task, global_cfg)
    content = _render_multilabel_script(
        job_name=job_name_seeded,
        job_slug=job_slug,
        data_path=data_path,
        model_dir=model_dir,
        keep_columns=keep_columns,
        global_cfg=global_cfg,
        epochs=epochs,
        chemprop_seed=chemprop_seed,
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
        JobDefinition(job_name=job_name_seeded, filename=filename, content=content, submit=submit_flag)
    ]


def _build_regression_jobs(task: Mapping[str, object], global_cfg: GC, renderers: Renderers) -> list[JobDefinition]:
    _reject_overrides(task, ["split_column", "score_column", "score_columns"])
    job_name_base = _extract_job_name(task)
    data_path = _task_data_path(task, global_cfg)
    task_type = str(task.get("task_type", "regression"))
    split_seed_value = int(task["split_seed"])
    chemprop_seed = int(task.get("chemprop_seed", split_seed_value))

    score_list = [_ensure_seed_suffix("score", split_seed_value)]

    epochs = int(task.get("epochs", global_cfg.epochs))
    split_column = _split_column(split_seed_value)
    ensemble_size = int(task.get("ensemble_size", 1))
    compound_min = task.get("compound_min_screens")
    compound_min_value = float(compound_min) if compound_min is not None else None
    compound_screens_column = task.get("compound_screens_column", "screens")
    if compound_screens_column is None:
        compound_screens_column = "screens"
    compound_screens_column = str(compound_screens_column)
    screens_weight_mode = _screens_weight_mode_from_task(task)
    submit_flag = _should_submit(task, global_cfg)

    job_defs: list[JobDefinition] = []
    for score_col in score_list:
        target_slug = _slugify(score_col)
        job_name_seeded = _ensure_seed_suffix(job_name_base, split_seed_value)
        per_job_name = job_name_seeded if len(score_list) == 1 else f"{job_name_seeded}_{target_slug}"
        per_slug = _slugify(per_job_name)
        context = {
            "job_name": per_job_name,
            "job_slug": per_slug,
            "target": target_slug,
            "target_column": score_col,
        }
        model_dir = _format_model_dir(task.get("model_dir"), global_cfg, context)

        content = _render_regression_script(
            job_name=per_job_name,
            job_slug=per_slug,
            data_path=data_path,
            model_dir=model_dir,
            target_column=score_col,
            task_type=task_type,
            global_cfg=global_cfg,
            epochs=epochs,
            chemprop_seed=chemprop_seed,
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


def _build_threshold_jobs(task: Mapping[str, object], global_cfg: GC, renderers: Renderers) -> list[JobDefinition]:
    _reject_overrides(task, ["split_column", "score_column", "score_columns"])
    job_name_base = _extract_job_name(task)
    data_path = _task_data_path(task, global_cfg)
    split_seed_value = int(task["split_seed"])
    chemprop_seed = int(task.get("chemprop_seed", split_seed_value))
    metric_column = _ensure_seed_suffix("score", split_seed_value)
    target_column = str(task.get("target_column", "target"))
    epochs = int(task.get("epochs", global_cfg.epochs))
    split_column = _split_column(split_seed_value)
    ensemble_size = int(task.get("ensemble_size", 1))
    compound_min = task.get("compound_min_screens")
    compound_min_value = float(compound_min) if compound_min is not None else None
    compound_screens_column = task.get("compound_screens_column", "screens")
    if compound_screens_column is None:
        compound_screens_column = "screens"
    compound_screens_column = str(compound_screens_column)
    screens_weight_mode = _screens_weight_mode_from_task(task)
    submit_flag = _should_submit(task, global_cfg)

    threshold_specs = task.get("thresholds")
    if not isinstance(threshold_specs, list) or not threshold_specs:
        raise ConfigError("Threshold tasks require a non-empty thresholds list.")

    jobs: list[JobDefinition] = []
    for idx, spec in enumerate(threshold_specs):
        if not isinstance(spec, Mapping):
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

        job_name_seeded = _ensure_seed_suffix(job_name_base, split_seed_value)
        per_job_name = f"{job_name_seeded}_{suffix_str}"
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

        content = _render_threshold_script(
            job_name=per_job_name,
            job_slug=per_slug,
            data_path=data_path,
            model_dir=model_dir,
            metric_column=metric_column,
            target_column=target_column,
            global_cfg=global_cfg,
            epochs=epochs,
            chemprop_seed=chemprop_seed,
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


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _task_metrics(task: Mapping[str, object], key: str, defaults: list[str]) -> list[str]:
    metrics = task.get(key)
    if metrics is None:
        return defaults
    return [str(m) for m in metrics]


def _reject_overrides(task: Mapping[str, object], keys: list[str]) -> None:
    for key in keys:
        if key in task:
            raise ConfigError(f"'{key}' is not configurable; remove it from the task.")


def _regression_metrics(task: Mapping[str, object], global_cfg: BaseGlobalConfig) -> list[str]:
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


CLI_MODULE = "model_jobs_uge.cli"


def _render_multilabel_script(
    *,
    job_name: str,
    job_slug: str,
    data_path: Path,
    model_dir: Path,
    keep_columns: list[str],
    global_cfg: BaseGlobalConfig,
    epochs: int,
    chemprop_seed: int,
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
        f'"${{PYTHON_BIN}}" -m {CLI_MODULE} trim-columns \\',
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
        train_cmd_lines.append(f"  --pytorch-seed {chemprop_seed} \\")
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
    chemprop_seed: int,
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
        f'"${{PYTHON_BIN}}" -m {CLI_MODULE} prepare-regression \\',
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
        train_cmd_lines.append(f"  --pytorch-seed {chemprop_seed} \\")
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
    chemprop_seed: int,
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
        f'"${{PYTHON_BIN}}" -m {CLI_MODULE} prepare-threshold-classifier \\',
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
        train_cmd_lines.append(f"  --pytorch-seed {chemprop_seed} \\")
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
# UGE rendering
# ---------------------------------------------------------------------------


def _render_header(job_name: str, global_cfg: GlobalConfig) -> str:
    lines: list[str] = ["#!/bin/bash"]
    lines.append(f"#$ -N {job_name}")
    lines.append(f"#$ -pe smp {global_cfg.cpus}")
    lines.append("#$ -cwd")
    lines.append("#$ -S /bin/bash")
    lines.append(f"#$ -l m_mem_free={global_cfg.memory}")
    lines.append(f"#$ -l h_rt={global_cfg.runtime}")

    if global_cfg.gpu_cards > 0:
        lines.append(f"#$ -l gpu_card={global_cfg.gpu_cards}")

    log_prefix = global_cfg.logs_dir / job_name
    lines.append(f"#$ -e {log_prefix}_$JOB_ID.err")
    lines.append(f"#$ -o {log_prefix}_$JOB_ID.out")

    for directive in global_cfg.queue_directives:
        formatted = str(directive).strip()
        if formatted:
            lines.append(f"#$ {formatted}")

    if global_cfg.email and global_cfg.email_notifications:
        lines.append(f"#$ -M {global_cfg.email}")
        lines.append(f"#$ -m {global_cfg.notify_events}")

    for extra in global_cfg.extra_header_directives:
        extra = str(extra).strip()
        if extra:
            lines.append(extra)

    return "\n".join(lines)


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
    if global_cfg.gpu_cards > 0 and global_cfg.gpu_map_script:
        lines.append("")
        lines.append(f"export CUDA_VISIBLE_DEVICES=$({global_cfg.gpu_map_script})")
        lines.append('echo "[INFO] [$START] [$(date)] [$$] Selected GPUs: ${CUDA_VISIBLE_DEVICES}"')
    _append_conda_and_binaries(lines, global_cfg)
    return "\n".join(lines)


_UGE_RENDERERS = Renderers(
    header=_render_header,
    common_env=_render_common_env,
)
