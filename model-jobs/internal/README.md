Internal configuration and helpers
==================================

This directory contains **internal-only** configuration files and helpers used
to run experiments on the private UGE-based cluster. Everything under
`model_jobs/internal/` is safe to delete when preparing a public release of
the code accompanying the paper.

Contents
--------

- `configs/`: YAML submission plans that reference internal paths, email
  addresses, queue parameters, and `qsub`. These are meant to be consumed by
  the UGE-aware job generator in `model_jobs.jobgen_uge`.

Usage
-----

On the internal cluster, point your submission commands at the internal module,
which generates UGE headers and submits via `qsub`, for example:

```bash
python -m model_jobs.cli_uge \
  --config internal/configs/main_experiments.yaml \
  --output-dir scripts/generated
```

For public use, rely on the standard `model-jobs submit-jobs` CLI and the
public configuration files under `model_jobs/configs/`, which are free of
cluster-specific settings. The internal modules depend on the public helpers,
so you can delete `model_jobs/internal/`, `model_jobs/jobgen_uge.py`,
and `model_jobs/cli_uge.py` before publishing the open-source release.

Files to remove for the public release:
- `model_jobs/internal/`
- `model_jobs/src/model_jobs/jobgen_uge.py`
- `model_jobs/src/model_jobs/cli_uge.py`
