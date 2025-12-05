from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Sequence

import typer

from model_jobs.cli import _setup_logging, _write_scripts_and_maybe_submit, PROGRESS_COLUMNS
from model_jobs.jobgen_uge import ConfigError, load_submission_plan

LOGGER = logging.getLogger("model_jobs.cli_uge")

uge_app = typer.Typer(
    no_args_is_help=True,
    help="Generate UGE-ready Chemprop scripts and submit them via qsub.",
)


@uge_app.callback()
def _main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    _setup_logging(verbose)


@uge_app.command("submit-jobs")
def submit_jobs(
    config_path: Path = typer.Option(..., "--config", help="Path to the submission YAML file."),
    output_dir: Path = typer.Option(
        ..., "--output-dir", help="Directory for generated shell scripts."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Generate scripts without invoking qsub."
    ),
    submit_override: bool | None = typer.Option(
        None,
        "--submit/--no-submit",
        help="Force submit or skip execution regardless of config.",
    ),
) -> None:
    """Generate UGE-ready Chemprop scripts and optionally submit them via qsub."""
    try:
        global_cfg, jobs = load_submission_plan(config_path)
    except ConfigError as exc:
        raise typer.BadParameter(str(exc))

    for path in (output_dir, global_cfg.logs_dir, global_cfg.models_dir, global_cfg.temp_dir):
        path.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loaded %d job(s) from %s", len(jobs), config_path)

    def _submit_with_qsub(script_path: Path) -> None:
        cmd = ["qsub", str(script_path)]
        LOGGER.info("Submitting %s via %s", script_path, " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError as exc:
            raise typer.Exit(1) from exc
        except subprocess.CalledProcessError as exc:
            raise typer.Exit(exc.returncode) from exc

    _write_scripts_and_maybe_submit(
        jobs=jobs,
        output_dir=output_dir,
        submit_override=submit_override,
        dry_run=dry_run,
        submitter=_submit_with_qsub,
        submit_label="qsub submission",
        logger=LOGGER,
    )


def main() -> None:
    uge_app()


if __name__ == "__main__":
    main()
