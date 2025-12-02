# train-pred

CLI utilities for preparing Chemprop training and prediction datasets and generating job scripts.

- Python package: `train-pred` (module `train_pred`)
- CLI entry point: `train-pred`

The main user-facing interface is the `submit-jobs` subcommand, which reads a YAML config and emits Chemprop training/prediction shell scripts:

```bash
python -m train_pred.cli submit-jobs \
  --config configs/example_jobs.yaml \
  --output-dir scripts/generated \
  --dry-run
```

On internal clusters you can also use the UGE-specific helpers described in `train-pred/internal/README.md`.

