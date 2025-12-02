Internal configuration and helpers
==================================

This directory contains **internal-only** configuration files and helpers used
to run experiments on the private UGE-based cluster. Everything under
`train-pred/internal/` is safe to delete when preparing a public release of
the code accompanying the paper.

Contents
--------

- `configs/`: YAML submission plans that reference internal paths, email
  addresses, queue parameters, and `qsub`. These are meant to be consumed by
  the UGE-aware job generator in `train_pred.jobgen_uge`.

Usage
-----

On the internal cluster, point your submission commands at the internal module,
which generates UGE headers and submits via `qsub`, for example:

```bash
python -m train_pred.cli_uge \
  --config internal/configs/main_experiments.yaml \
  --output-dir scripts/generated
```

For public use, rely on the standard `train-pred submit-jobs` CLI and the
public configuration files under `train-pred/configs/`, which are free of
cluster-specific settings. The internal modules depend on the public helpers,
so you can delete `train-pred/internal/`, `train_pred/jobgen_uge.py`,
and `train_pred/cli_uge.py` before publishing the open-source release.

Files to remove for the public release:
- `train-pred/internal/`
- `train-pred/src/train_pred/jobgen_uge.py`
- `train-pred/src/train_pred/cli_uge.py`
