# Model jobs

CLI utilities for preparing Chemprop training/prediction datasets and generating shell scripts.

Quick start:

```bash
python -m model_jobs.cli write-scripts \
  --config configs/examples/local_grid.yaml \
  --output-dir scripts/generated
```

Config outline (see `configs/examples/local_grid.yaml`):
- `global`: paths, CPU count, optional Conda activation, dataset aliases (keys referenced by tasks).
- `defaults`: shared arguments for all tasks plus per-type blocks (`multilabel`, `regression`, `threshold`), and an optional `hit_rate` block applied to both regression and threshold tasks.
  Strings can use `{variables}` pulled from expansions (e.g., `split_seed: "{seed}"`).
- `sweeps`: reusable lists (e.g., seeds, percentiles). Refer to them with `@name`.
- `tasks`: each entry has a `type` and optional `expand` block that explodes the task across sweep values.
  Threshold tasks can use `thresholds: {expand: ..., template: ...}` to generate many cutoffs without a
  helper script.
  - Common task fields: `data_path` (alias key from `global.datasets` or absolute path). Every task must provide `split_seed` (directly or via defaults/expand); the split column and score column are fixed to `split_seed{N}` and `score_seed{N}` for that seed, and the job name is suffixed automatically. `chemprop_seed` (optional) controls the training RNG seed (defaults to `split_seed` if omitted).

Generated scripts live under `--output-dir` and are executable; run them manually on your scheduler/host.
