# Frequent hitter prediction workspace

This repository is organized by stage so new readers can follow the flow end-to-end:

1. `assay-etl` – download, annotate, and preprocess PubChem BioAssay tables.
2. `assay-cleaning` – clean HTS outputs and split assays into biochemical / cellular sets.
3. `dataset-pipeline` – scaffold-aware dataset builder that turns cleaned hits into train/val/calibration/test artifacts.
4. `model-jobs` – CLI helpers to generate training/prediction scripts (with an internal UGE variant).
5. `chemprop` – upstream fork providing model training; kept as a separate submodule and not modified here.

## Environment setup

To create a shared conda environment named `frequent-hitters` that can run all of the core components in this repo (pipelines, Chemprop integration, and PubChem ETL without LLM backends), run:

```bash
conda env create -f environment.frequent-hitters.yml
conda activate frequent-hitters

# Install PyTorch Lightning and PyTorch with the correct CUDA version

# Install local packages without pulling in extra deps:
cd chemprop && python -m pip install --no-deps .
cd ../dataset-pipeline && python -m pip install --no-deps .
cd ../model-jobs && python -m pip install --no-deps .
cd ../assay-cleaning && python -m pip install --no-deps .
cd ../assay-etl && python -m pip install --no-deps .
```
