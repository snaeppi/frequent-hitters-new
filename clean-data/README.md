# clean-data

Tools for cleaning HTS data and splitting it into biochemical and cellular subsets using Hit Dexter 3–style curation.

- Python package: `clean-split` (module `clean_split`)
- CLI entry point: `clean-split`
- Legacy script shim: `clean-data/clean_and_split.py`

Example usage:

```bash
python -m clean_split.cli \
  --hts-file path/to/assay_rscores.parquet \
  --assay-props-file path/to/assay_metadata.csv \
  --biochemical-out biochemical_hits.parquet \
  --cellular-out cellular_hits.parquet \
  --rename-cols
```

