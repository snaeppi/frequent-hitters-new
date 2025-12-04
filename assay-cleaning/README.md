# assay_cleaning

Tools for cleaning HTS data and splitting it into biochemical and cellular subsets using Hit Dexter 3–style curation.

- Python package: `assay-cleaning` (module `assay_cleaning`)
- CLI entry point: `assay-cleaning`

Example usage:

```bash
python -m assay_cleaning.cli \
  --hts-file path/to/assay_rscores.parquet \
  --assay-props-file path/to/assay_metadata.csv \
  --biochemical-out biochemical_hits.parquet \
  --cellular-out cellular_hits.parquet \
  --rename-cols
```
