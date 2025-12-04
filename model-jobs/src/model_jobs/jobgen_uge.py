from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .jobgen import (
    ConfigError,
    JobDefinition,
    Renderers,
    _append_conda_and_binaries,
    _common_env_base_lines,
    _load_submission_plan,
    _resolve_path,
)

__all__ = ["ConfigError", "load_submission_plan", "JobDefinition", "GlobalConfig"]


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
    seed: int
    notify_events: str
    email_notifications: bool
    gpu_map_script: str | None
    metrics_classification: list[str]
    metrics_regression: list[str]
    chemprop_train_cmd: str
    chemprop_predict_cmd: str
    submit: bool
    dataset_aliases: dict[str, Path] = field(default_factory=dict)
    extra_header_directives: list[str] = field(default_factory=list)
    base_path: Path = field(default_factory=Path)


def load_submission_plan(config_path: Path) -> tuple[GlobalConfig, list[JobDefinition]]:
    return _load_submission_plan(
        config_path,
        parse_global_config=_parse_global_config,
        renderers=_RENDERERS,
    )


def _parse_global_config(raw: dict, base_path: Path) -> GlobalConfig:
    if "models_dir" not in raw:
        raise ConfigError("global.models_dir is required.")
    if "temp_dir" not in raw:
        raise ConfigError("global.temp_dir is required.")

    dataset_aliases = {}
    for key, value in (raw.get("datasets") or {}).items():
        dataset_aliases[str(key)] = _resolve_path(value, base_path)

    queue_directives = raw.get("queue_directives") or ["-j y", "-R y", "-notify"]

    email = str(raw.get("email", ""))
    logs_raw = raw.get("logs_dir")
    logs_dir = _resolve_path(logs_raw, base_path) if logs_raw is not None else (base_path / "logs").resolve()

    return GlobalConfig(
        email=email,
        logs_dir=logs_dir,
        models_dir=_resolve_path(raw["models_dir"], base_path),
        temp_dir=_resolve_path(raw["temp_dir"], base_path),
        cpus=int(raw.get("cpus", raw.get("cores", 1))),
        memory=str(raw.get("memory", "12G")),
        runtime=str(raw.get("runtime", "100000")),
        gpu_cards=int(raw.get("gpu_cards", raw.get("gpu", 1))),
        queue_directives=[str(d) for d in queue_directives],
        conda_commands=[str(cmd) for cmd in raw.get("conda_commands", [])],
        conda_activate=str(raw.get("conda_activate", "")),
        python_executable=str(raw.get("python", "python")),
        epochs=int(raw.get("epochs", 50)),
        seed=int(raw.get("seed", 42)),
        notify_events=str(raw.get("notify_events", "bea")),
        email_notifications=bool(raw.get("email_notifications", True)),
        gpu_map_script=str(raw.get("gpu_map_script")) if raw.get("gpu_map_script") else None,
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
        extra_header_directives=[str(d) for d in raw.get("extra_directives", [])],
        base_path=base_path,
    )


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
        formatted = _format_directive(directive)
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


def _format_directive(directive: str) -> str:
    directive = str(directive).strip()
    return directive


def _render_common_env(global_cfg: GlobalConfig, job_name: str) -> str:
    lines = _common_env_base_lines(global_cfg, job_name)
    if global_cfg.gpu_cards > 0 and global_cfg.gpu_map_script:
        lines.append("")
        lines.append(f'export CUDA_VISIBLE_DEVICES=$({global_cfg.gpu_map_script})')
        lines.append('echo "[INFO] [$START] [$(date)] [$$] Selected GPUs: ${CUDA_VISIBLE_DEVICES}"')
    _append_conda_and_binaries(lines, global_cfg)
    return "\n".join(lines)


_RENDERERS = Renderers(
    header=_render_header,
    common_env=_render_common_env,
)
