# Code for frequent hitter prediction

This codebase contains:

1. `pubchem-bioassay`, a collection of scripts for downloading, annotating, and preprocessing PubChem Bioassay data.
2. `clean-data`, which contains the `clean_split` tooling for cleaning both the proprietary internal data and public PubChem Bioassay data. At the moment there remain references to internal data in the form of defaults set in the argparser. These should be removed, possibly in favor of a config file so we can easily have separate config for internal and pubchem data.
3. `data-pipeline`, which includes the code to process the cleaned high-throughput screening data into the training, validation, calibration, and testing data for the different types of frequent hitter prediction models.
4. `chemprop`, a fork of Chemprop that provides extra features such as parquet I/O and class weight balancing for binary classifiers.
5. `train-pred`, a CLI tool for generating the training and prediction scripts for running the experiments.

## Environment setup

To create a shared conda environment named `frequent-hitters` that can run all of the core components in this repo (pipelines, Chemprop integration, and PubChem ETL without LLM backends), run:

```bash
conda env create -f environment.frequent-hitters.yml
conda activate frequent-hitters

# Install local packages without pulling in extra deps:
cd chemprop && python -m pip install --no-deps .
cd ../data-pipeline && python -m pip install --no-deps .
cd ../train-pred && python -m pip install --no-deps .
cd ../clean-data && python -m pip install --no-deps .
```

LLM-based annotation support for `pubchem-bioassay` can be installed separately (for example on a development machine rather than the internal HPC cluster):

```bash
python -m pip install "langchain" "langchain-openai" "langchain-google-vertexai" "langchain-xai"
```
