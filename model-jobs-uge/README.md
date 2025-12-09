# Model jobs UGE

UGE-specific Chemprop job generator. Scripts include UGE headers and can be
auto-submitted with `qsub` or simply written to disk for manual use.

- Python package: `model_jobs_uge` (installed as `model-jobs-uge`)
- CLI entry point: `model-jobs-uge`

Usage:

```bash
python -m model_jobs_uge.cli submit-jobs \
  --config configs/examples/uge_grid.yaml \
  --output-dir scripts/generated \
  --dry-run           # write scripts only
# add --no-submit to force skip qsub even if the config has submit: true
```

Config shape matches the public package (global/defaults/sweeps/tasks with `expand`),
with extra required UGE fields under `global`:
- `email`, `logs_dir`, `models_dir`, `temp_dir`
- `cpus`, `memory`, `runtime`, `gpu_cards`
- optional `queue_directives`, `notify_events`, `gpu_map_script`, `extra_directives`
- Tasks use `data_path` for inputs. Every task must provide `split_seed`; split/score columns and job names are automatically suffixed with `_seed{N}` to keep them aligned. Optional `chemprop_seed` controls the training RNG (defaults to `split_seed`).

See `configs/examples/uge_grid.yaml` for a templated multi-seed setup.
