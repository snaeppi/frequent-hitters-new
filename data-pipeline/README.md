# data-pipeline

Murcko-scaffold–aware pipeline for turning cleaned HTS data into model-ready datasets (train/validation/calibration/test) for frequent hitter prediction.

- Python package: `pipeline`
- Hydra entry point: `python -m pipeline.cli`

The pipeline is configured via Hydra YAML configs under `data-pipeline/configs/`. A typical invocation mirrors the integration test:

```bash
python -m pipeline.cli \
  assay_format=both \
  "paths.output_root=outputs" \
  "paths.input.biochemical=path/to/biochemical_hits.parquet" \
  "paths.input.cellular=path/to/cellular_hits.parquet"
```

