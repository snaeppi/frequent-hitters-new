from __future__ import annotations

import argparse
import logging
import os
import shlex
import subprocess
from pathlib import Path
from typing import Callable, Sequence

from train_pred.cli import _setup_logging
from train_pred.jobgen_uge import ConfigError, load_submission_plan

LOGGER = logging.getLogger("train_pred.cli_uge")


def _configure_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train-pred-qsub",
        description="Generate UGE-ready Chemprop scripts and submit them via qsub.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    parser.add_argument("--config", required=True, help="Path to the submission YAML file.")
    parser.add_argument("--output-dir", required=True, help="Directory for generated shell scripts.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate scripts without invoking qsub.",
    )
    parser.add_argument(
        "--qsub-command",
        help="Override the qsub command defined in the YAML (defaults to global.qsub_command).",
    )
    submit_group = parser.add_mutually_exclusive_group()
    submit_group.add_argument(
        "--submit",
        action="store_const",
        const=True,
        dest="submit_override",
        help="Submit all generated scripts regardless of config settings.",
    )
    submit_group.add_argument(
        "--no-submit",
        action="store_const",
        const=False,
        dest="submit_override",
        help="Do not submit scripts (overrides config).",
    )
    parser.set_defaults(submit_override=None)
    parser.set_defaults(handler=_handle_submit_jobs)
    return parser


def _handle_submit_jobs(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)

    try:
        global_cfg, jobs = load_submission_plan(config_path)
    except ConfigError as exc:
        LOGGER.error("Invalid submission configuration: %s", exc)
        raise SystemExit(1) from exc

    for path in (output_dir, global_cfg.logs_dir, global_cfg.models_dir, global_cfg.temp_dir):
        path.mkdir(parents=True, exist_ok=True)

    override = args.submit_override
    dry_run = bool(args.dry_run)
    qsub_command_raw = args.qsub_command or global_cfg.qsub_command
    qsub_command_parts = shlex.split(str(qsub_command_raw))
    if not qsub_command_parts:
        raise SystemExit("qsub command is empty; provide --qsub-command or set global.qsub_command")

    LOGGER.info("Loaded %d job(s) from %s", len(jobs), config_path)

    for job in jobs:
        script_path = output_dir / job.filename
        content = job.content if job.content.endswith("\n") else f"{job.content}\n"
        script_path.write_text(content)
        os.chmod(script_path, 0o750)

        should_submit = job.submit if override is None else override

        if dry_run:
            LOGGER.info("[DRY-RUN] Generated %s (submission skipped)", script_path)
            continue

        if should_submit:
            cmd = qsub_command_parts + [str(script_path)]
            LOGGER.info("Submitting %s via %s", script_path, shlex.join(cmd))
            try:
                subprocess.run(cmd, check=True)
            except FileNotFoundError as exc:
                LOGGER.error("qsub command not found: %s", qsub_command_parts[0])
                raise SystemExit(1) from exc
            except subprocess.CalledProcessError as exc:
                LOGGER.error("qsub failed for %s with exit code %s", script_path, exc.returncode)
                raise SystemExit(exc.returncode) from exc
        else:
            LOGGER.info("Generated %s (submission skipped by config)", script_path)

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _configure_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    handler: Callable[[argparse.Namespace], object] = args.handler
    result = handler(args)
    if isinstance(result, int):
        return result
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
