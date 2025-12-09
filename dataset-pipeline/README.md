# Dataset pipeline

 Murcko-scaffold–aware pipeline for turning cleaned HTS data into model-ready datasets (train/validation/test) for frequent hitter prediction.

- Python package: `dataset_pipeline`
- Hydra entry point: `python -m dataset_pipeline.cli`

The pipeline is configured via Hydra YAML configs under `dataset_pipeline/config/`. A typical invocation mirrors the integration test:

```bash
python -m dataset_pipeline.cli \
  assay_format=both \
  "paths.output_root=outputs" \
  "paths.input.biochemical=path/to/biochemical_hits.parquet" \
  "paths.input.cellular=path/to/cellular_hits.parquet" \
  "split.seeds=[1,2,3]"
```

Outputs per assay format:
- Unified regression dataset: `<format>_regression.parquet`
- Unified multi-task dataset: `<format>_multilabel.parquet`
- Threshold metadata: `<format>_thresholds.json`
- Split columns per seed: `split_seed<seed>` in each dataset (train/val/test). The regression and multi-task files share the same test compounds per seed; the regression file contains only regression-eligible rows.
