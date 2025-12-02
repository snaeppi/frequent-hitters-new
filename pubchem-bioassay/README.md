# PubChem Bioassay Preprocessing Tool

This module contains a file‑based preprocessing pipeline for PubChem bioassay data.  
It downloads assay data tables from PubChem, aggregates them to one row per compound,  
builds assay‑level metadata, supports manual column selection, LLM‑assisted annotation,  
and exports r‑scores for downstream machine learning.

All assay‑level metadata is stored in a small SQLite database under `outputs/`,  
with CSV only used as an exported view.

---

## Installation

If you use `uv`, from the `pubchem-bioassay` directory:

```bash
uv sync             # install dependencies from pyproject.toml / uv.lock
uv run pubchem-bioassay --help
```

To enable LLM-based annotation when using `uv`, include the `llm-annotation` extra:

```bash
uv sync --extra llm-annotation
uv run pubchem-bioassay annotate-metadata --help
```

Or with plain `pip`, from `pubchem-bioassay`:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install .
pubchem-bioassay --help
```

To add support for LLM-based annotation with `pip`, install the extra:

```bash
pip install .[llm-annotation]
pubchem-bioassay annotate-metadata --help
```

You can also run via `python -m`:

```bash
PYTHONPATH=src python -m pubchem_bioassay.cli.main --help
```

---

## Input files

The tool expects these inputs (already present in `minimal/data`):

- `data/pcassay_result.txt`  
  Text export of the pcassay catalog used to discover AIDs.

- `data/hd3_annotations.csv`  
  Hit Dexter 3.0 annotations (semicolon‑separated) with columns like `"data set"` and `"AID"`.
  Used to derive:
  - `target_type` / `bioactivity_type`
  - `assay_format` (`cellular`, `biochemical`, or `other`)

---

## Overview of the pipeline

Typical flow for one or many assays:

1. Download raw assay tables from PubChem (`download-assays`).
2. Aggregate to one row per `PUBCHEM_CID` (`aggregate-compounds`).
3. Build assay‑level metadata (`summarize-assays` → SQLite DB, optional CSV export).
4. Manually select a primary screening column (`select-column`)  
   or pre-fill `selected_column` in the DB (via `import-metadata-csv`) and then compute stats (`update-selected-stats`).
5. Optionally, run LLM-based annotation to fill in missing `target_type` / `bioactivity_type` (`annotate-metadata`).
6. Compute r‑scores and export a combined parquet (`compute-rscores`).

All commands are subcommands of `pubchem-bioassay`.

---

## Commands

### 1. Download assay tables – `download-assays`

Fetches compressed CSVs from PubChem and materializes them as Parquet.

```bash
pubchem-bioassay download-assays [OPTIONS]
```

Key options:

- `--aid AID`  
  Download a single AID.

- `--from-pcassay`  
  Download all AIDs discovered in `data/pcassay_result.txt`.

- `--aids-file PATH`  
  Optional file with one AID per line.

- `--pcassay-result PATH` (default `data/pcassay_result.txt`)

- `--out-dir PATH` (default `data/assay_tables`)  
  Where `aid_<AID>.parquet` will be written.

- `--force-download/--no-force-download`  
  Re‑download even if the parquet already exists.

Output:

- `data/assay_tables/aid_<AID>.parquet` – raw assay table for each AID.

Example (single AID):

```bash
pubchem-bioassay download-assays --aid 588334
```

### 2. Aggregate to one row per compound – `aggregate-compounds`

Aggregates each assay table to one row per `PUBCHEM_CID` using median (numeric)  
and mode (categorical).

```bash
pubchem-bioassay aggregate-compounds [OPTIONS]
```

Key options:

- `--aid AID`  
  Aggregate a single AID.

- `--from-pcassay` / `--aids-file PATH` / `--pcassay-result PATH`  
  Same semantics as `download-assays`.

- `--raw-dir PATH` (default `data/assay_tables`)  
  Location of `aid_<AID>.parquet` input files.

- `--out-dir PATH` (default `outputs/aggregated`)  
  Destination for `aid_<AID>_cid_agg.parquet`.

Output:

- `outputs/aggregated/aid_<AID>_cid_agg.parquet` – one row per compound.

Example:

```bash
pubchem-bioassay aggregate-compounds --aid 588334
```

### 3. Build assay metadata – `summarize-assays`

Creates or refreshes assay metadata in the SQLite DB, including Hit Dexter labels,  
and optionally exports a CSV snapshot.

```bash
pubchem-bioassay summarize-assays [OPTIONS]
```

Key options:

- `--pcassay-result PATH` (default `data/pcassay_result.txt`)
- `--hd3-annotations PATH` (default `data/hd3_annotations.csv`)
- `--metadata-db PATH` (default `outputs/assay_metadata.sqlite`)  
  SQLite database storing metadata (canonical store).
- `--metadata-csv PATH` (default `outputs/assay_metadata.csv`)  
  Optional CSV export of the current metadata table.

The underlying schema in SQLite and the CSV export share these columns:

- `aid`
- `target_type`
- `bioactivity_type`
- `assay_format`  
  - `"biochemical"` iff `target_type == "target-based"` and `bioactivity_type == "specific bioactivity"`.
  - `"cellular"` iff `target_type == "cell-based"` and `bioactivity_type == "specific bioactivity"`.
  - `"other"` for any other annotated case.
  - Empty if there is no Hit Dexter annotation for that AID.
- `selected_column` (initially empty)
- `median`, `mad`
- `compounds_screened`, `coverage`
- `hits_rscore`, `hits_overlap`, `hits_outcome`
- `rscore_hit_rate`

Example:

```bash
pubchem-bioassay summarize-assays
```

### 4. Interactive column selection – `select-column`

Interactive CLI to choose the primary screening column for one assay.  
This is the “annotation CLI tool” for column selection.

```bash
pubchem-bioassay select-column [OPTIONS] AID
```

Key options:

- `AID` (argument) – assay ID to annotate.

- `--raw-dir PATH` (default `data/assay_tables`)  
  Directory containing `aid_<AID>.parquet`.

- `--aggregated-dir PATH` (default `outputs/aggregated`)  
  Directory with `aid_<AID>_cid_agg.parquet`. If missing, it will be created on the fly from `raw-dir`.

- `--metadata-db PATH` (default `outputs/assay_metadata.sqlite`)  
  Metadata SQLite DB to update with `selected_column` and stats.

Behavior:

- Ensures an aggregated parquet exists for the AID.
- Detects candidate numeric columns (drops most `PUBCHEM_*` metadata).
- Detects replicate groups like `REPLICATE_A_...`, `REPLICATE_B_...` and writes mean columns such as `_ACTIVITY_SCORE_12.5uM_(%)_mean`.
- For each candidate column, computes:
  - `median`, `mad`
  - `compounds_screened` (row count)
  - `coverage` (proportion non‑null)
  - `hits_rscore` – count of compounds with `|r| ≥ 3`
  - `hits_overlap` – r‑score hits that are also outcome “Active”
  - `hits_outcome` – outcome‑based actives
- Shows a table and highlights the column with highest `hits_overlap` (ties broken by coverage).
- Prompts:
  - Enter index to pick a specific column.
  - Press Enter to pick the highlighted best.
  - `s` to skip.
  - `i` to mark the assay as ineligible (`selected_column="__INELIGIBLE__"`).
- Writes the choice and all stats into the metadata DB.

Example:

```bash
pubchem-bioassay select-column 588334
```

### 5. Interactive selection for all missing – `select-column-all`

Run the interactive selector for every assay that does **not** yet have a `selected_column` in the metadata DB.

```bash
pubchem-bioassay select-column-all [OPTIONS]
```

Key options:

- `--raw-dir PATH` (default `data/assay_tables`)  
  Directory containing `aid_<AID>.parquet`.

- `--aggregated-dir PATH` (default `outputs/aggregated`)  
  Directory containing or receiving `aid_<AID>_cid_agg.parquet`.

- `--metadata-db PATH` (default `outputs/assay_metadata.sqlite`)  
  Metadata DB to read/write selections from/to.

- `--start-aid INT` (optional)  
  Skip any AIDs below this numeric value.

Behavior:

- Looks up all `aid` where `selected_column` is `NULL` or empty in SQLite.
- Iterates them in ascending order, calling the same interactive UI as `select-column`.
- At any point you can stop with `Ctrl+C`; already processed AIDs remain stored.

### 5. Compute stats for existing selections – `update-selected-stats`

Use this when `selected_column` has been set manually (e.g., joined from an older pipeline)  
and you want to compute median/MAD and hit statistics for those columns.

```bash
pubchem-bioassay update-selected-stats [OPTIONS]
```

Key options:

- `--metadata-db PATH` (default `outputs/assay_metadata.sqlite`)  
  Must contain `aid` and `selected_column`.

- `--aggregated-dir PATH` (default `outputs/aggregated`)  
  Location of `aid_<AID>_cid_agg.parquet`.

Behavior:

- For each row where `selected_column` is non‑empty and not `"__INELIGIBLE__"`:
  - Loads the aggregated parquet for that AID.
  - Locates the candidate stats for the selected column.
  - Updates the metadata row in SQLite with:
    - `median`, `mad`
    - `compounds_screened`, `coverage`
    - `hits_rscore`, `hits_overlap`, `hits_outcome`
    - `rscore_hit_rate` (`hits_rscore / compounds_screened`)

This is the quickest way to “rehydrate” stats after manually populating `selected_column`.

### 6. Compute r‑scores and export – `compute-rscores`

Computes r‑scores for every assay with a selected column and writes a single parquet.

```bash
pubchem-bioassay compute-rscores [OPTIONS]
```

Key options:

- `--metadata-db PATH` (default `outputs/assay_metadata.sqlite`)  
  Must contain `aid`, `selected_column`, and optionally `median`, `mad`.

- `--aggregated-dir PATH` (default `outputs/aggregated`)  
  Directory with `aid_<AID>_cid_agg.parquet`.

- `--out PATH` (default `outputs/assay_rscores.parquet`)

Behavior:

- For each row where `selected_column` is non‑empty and not `"__INELIGIBLE__"`:
  - Loads the aggregated parquet.
  - If `median`/`mad` are missing, computes them on the fly from the selected column.
  - Computes `r_score = (value - median) / (mad * 1.4826)` (no further transforms).
  - Outputs one row per compound with:
    - `assay_id`, `compound_id`, `smiles`, `r_score`.
- Concatenates all assays into a single parquet at `--out`.

Example:

```bash
pubchem-bioassay compute-rscores
```

### 7. Import existing metadata CSV – `import-metadata-csv`

One-time migration helper to seed the SQLite metadata DB from an existing CSV  
(e.g., from an older run of the pipeline).

```bash
pubchem-bioassay import-metadata-csv [OPTIONS] [CSV_PATH]
```

Arguments / options:

- `CSV_PATH` (argument, default `outputs/assay_metadata.csv`)  
  CSV file to import.
- `--metadata-db PATH` (default `outputs/assay_metadata.sqlite`)  
  Destination metadata DB (will be created or overwritten logically row-by-row).

Use this once to migrate your existing `assay_metadata.csv` into SQLite before  
switching to the DB-based workflow.

### 8. Export metadata snapshot – `export-metadata-csv`

Export the current contents of the SQLite metadata table to a CSV for quick inspection.

```bash
pubchem-bioassay export-metadata-csv [OPTIONS]
```

Key options:

- `--metadata-db PATH` (default `outputs/assay_metadata.sqlite`)  
  Metadata DB to export from.
- `--out PATH` (default `outputs/assay_metadata.csv`)  
  Destination CSV path.

Use this after running `select-column` or `update-selected-stats` if you want an updated  
`assay_metadata.csv` for manual review or external analysis.

### 9. LLM-based annotation – `annotate-metadata`

Use a chat model to fill in `target_type` / `bioactivity_type` for assays that have a  
selected column but no existing annotation (and are not marked ineligible).

```bash
pubchem-bioassay annotate-metadata [OPTIONS]
```

Key options:

- `--aid INT` (optional)  
  Annotate a single AID; omit to process all eligible assays.

- `--metadata-db PATH` (default `outputs/assay_metadata.sqlite`)  
  Metadata DB containing assays and selections.

- `--model NAME` (default `gpt-4o-mini`)  
  OpenAI-compatible chat model name.

- `--max-hz FLOAT` (default `5.0`)  
  Maximum request rate to the PUG-REST server (via `RateLimiter`).

- `--out-dir PATH` (default `outputs/llm_annotations`)  
  Directory to write JSON files with the LLM classification + rationale and the parsed description.

Behavior:

- Finds assays where:
  - `selected_column` is set and not `__INELIGIBLE__`, and
  - both `target_type` and `bioactivity_type` are missing.
- For each such assay:
  - Fetches the description XML from PUG-REST.
  - Parses out name, source, description, and protocol.
  - Calls the LLM with clear definitions for Target Type and Bioactivity Type.
  - Writes the chosen labels into the metadata DB (and computes `assay_format` accordingly).
  - Saves a JSON file per AID in `out-dir` containing the labels, rationale, and description,  
    so you can review and manually override any incorrect annotation later.

---

## Example end‑to‑end for a single AID

Below is a concrete sequence for one assay (e.g. AID 588334):

```bash
cd minimal

# 1. Download raw assay table
pubchem-bioassay download-assays --aid 588334

# 2. Aggregate to one row per compound
pubchem-bioassay aggregate-compounds --aid 588334

# 3. Build or refresh metadata
pubchem-bioassay summarize-assays

# 4. Interactively select a column for this assay
pubchem-bioassay select-column 588334

# (Optionally repeat select-column for other AIDs.)

# 5. Compute r-scores for all assays with selected columns
pubchem-bioassay compute-rscores
```

After this, you will have:

- `outputs/assay_metadata.csv` – assay‑level metadata, including your selections and stats.
- `outputs/assay_rscores.parquet` – combined per‑compound r‑scores (`assay_id, compound_id, smiles, r_score`).
