# PubChem Assay ETL

This module contains a file‑based preprocessing pipeline for PubChem bioassay data.  
It downloads assay data tables from PubChem, aggregates them to one row per compound,  
builds assay‑level metadata, supports manual column selection and manual assay annotation,  
and exports r‑scores for downstream machine learning.

All assay‑level metadata is stored in a small SQLite database under `outputs/`,  
with CSV only used as an exported view.

---

## Installation

If you use `uv`, from the `assay_etl` directory:

```bash
uv sync             # install dependencies from pyproject.toml / uv.lock
uv run assay-etl --help
```

Or with plain `pip`, from `assay_etl`:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install .
assay-etl --help
```

You can also run via `python -m`:

```bash
PYTHONPATH=src python -m assay_etl.cli --help
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

1. Download and aggregate assay tables from PubChem (`download-assays`) – one parquet per AID with one row per `PUBCHEM_CID`.
2. Build assay‑level metadata (`summarize-assays` → SQLite DB, optional CSV export).
3. Manually select a primary screening column (`select-column`)  
   or pre-fill `selected_column` in the DB (via `import-metadata-csv`) and then compute stats (`update-selected-stats`).
4. Optionally, run manual annotation to fill in missing `target_type` / `bioactivity_type` (`annotate-metadata`) using assay descriptions and direct PubChem links.
5. Compute r‑scores and export a combined parquet (`compute-rscores`).

All commands are subcommands of `assay-etl`.

---

## Commands

### 1. Download assay tables – `download-assays`

Fetches compressed CSVs from PubChem, infers types, aggregates to one row per `PUBCHEM_CID`, and writes one Parquet per AID (atomic writes).

```bash
assay-etl download-assays [OPTIONS]
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
  Where aggregated `aid_<AID>.parquet` files will be written.

- `--force-download/--no-force-download`  
  Re‑download even if the parquet already exists.

- `--io-workers INT` (default 4)  
  Concurrent download threads.

- `--cpu-workers INT` (default 4)  
  Concurrent processing workers for CSV → aggregated parquet.

Output:

- `data/assay_tables/aid_<AID>.parquet` – aggregated per-assay table (one row per compound).

Example (single AID):

```bash
assay-etl download-assays --aid 588334
```

### 2. Build assay metadata – `summarize-assays`

Creates or refreshes assay metadata in the SQLite DB, including Hit Dexter labels,  
and optionally exports a CSV snapshot.

```bash
assay-etl summarize-assays [OPTIONS]
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
assay-etl summarize-assays
```

### 3. Interactive column selection – `select-column`

Interactive CLI to choose the primary screening column for one or many assays.  
Use `--aid` for a single assay, or `--from-pcassay` / `--aids-file` to iterate with a progress bar.

```bash
assay-etl select-column [OPTIONS]
```

Key options:

- `--aid INT`  
  Annotate a single assay.

- `--from-pcassay` / `--aids-file PATH` / `--pcassay-result PATH`  
  Iterate over AIDs discovered in the pcassay catalog or an explicit list.

- `--start-aid INT`  
  Skip any AIDs below this numeric value when batching.

- `--assays-dir PATH` (default `data/assay_tables`)  
  Directory containing aggregated `aid_<AID>.parquet` files (output of `download-assays`). `--aggregated-dir` is an alias.

- `--metadata-db PATH` (default `outputs/assay_metadata.sqlite`)  
  Metadata SQLite DB to update with `selected_column` and stats.

Behavior:

- Loads the aggregated parquet for the AID (one row per compound).
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

Example (single):

```bash
assay-etl select-column --aid 588334
```

Example (batch):

```bash
assay-etl select-column --from-pcassay --start-aid 500000
```

### 4. Compute stats for existing selections – `update-selected-stats`

Use this when `selected_column` has been set manually (e.g., joined from an older pipeline)  
and you want to compute median/MAD and hit statistics for those columns.

```bash
assay-etl update-selected-stats [OPTIONS]
```

Key options:

- `--metadata-db PATH` (default `outputs/assay_metadata.sqlite`)  
  Must contain `aid` and `selected_column`.

- `--aggregated-dir PATH` (default `data/assay_tables`)  
  Location of aggregated `aid_<AID>.parquet` tables (`--assays-dir` alias).

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

### 5. Compute r‑scores and export – `compute-rscores`

Computes r‑scores for every assay with a selected column and writes a single parquet.

```bash
assay-etl compute-rscores [OPTIONS]
```

Key options:

- `--metadata-db PATH` (default `outputs/assay_metadata.sqlite`)  
  Must contain `aid`, `selected_column`, and optionally `median`, `mad`.

- `--aggregated-dir PATH` (default `data/assay_tables`)  
  Directory with aggregated `aid_<AID>.parquet` files (`--assays-dir` alias).

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
assay-etl compute-rscores
```

### 6. Import existing metadata CSV – `import-metadata-csv`

One-time migration helper to seed the SQLite metadata DB from an existing CSV  
(e.g., from an older run of the pipeline).

```bash
assay-etl import-metadata-csv [OPTIONS] [CSV_PATH]
```

Arguments / options:

- `CSV_PATH` (argument, default `outputs/assay_metadata.csv`)  
  CSV file to import.
- `--metadata-db PATH` (default `outputs/assay_metadata.sqlite`)  
  Destination metadata DB (will be created or overwritten logically row-by-row).

Use this once to migrate your existing `assay_metadata.csv` into SQLite before  
switching to the DB-based workflow.

### 7. Export metadata snapshot – `export-metadata-csv`

Export the current contents of the SQLite metadata table to a CSV for quick inspection.

```bash
assay-etl export-metadata-csv [OPTIONS]
```

Key options:

- `--metadata-db PATH` (default `outputs/assay_metadata.sqlite`)  
  Metadata DB to export from.
- `--out PATH` (default `outputs/assay_metadata.csv`)  
  Destination CSV path.

Use this after running `select-column` or `update-selected-stats` if you want an updated  
`assay_metadata.csv` for manual review or external analysis.

### 8. Manual metadata annotation – `annotate-metadata`

Interactively assign `target_type` / `bioactivity_type` for assays with a selected column but no existing annotation (and not marked ineligible). Descriptions are shown in a pager so you can scroll before choosing labels.

```bash
assay-etl annotate-metadata [OPTIONS]
```

Key options:

- `--from-pcassay`  
  Iterate over assays from `pcassay_result.txt` (or `--aids-file`) that have a `selected_column` and are missing `target_type` or `bioactivity_type`.

- `--aid INT` (optional)  
  Annotate a single AID (even if already annotated, to override Hit Dexter labels).

- `--aids-file PATH` (optional)  
  Provide specific AIDs (one per line) instead of the entire pcassay catalog.

- `--pcassay-result PATH` (default `data/pcassay_result.txt`)  
  Source catalog when using `--from-pcassay`.

- `--metadata-db PATH` (default `outputs/assay_metadata.sqlite`)  
  Metadata DB containing assays and selections.

- `--start-aid INT` (optional)  
  Skip any AIDs below this numeric value when iterating all (use with `--from-pcassay`).

Behavior:

- Finds assays where:
  - `selected_column` is set and not `__INELIGIBLE__`, and
  - either `target_type` or `bioactivity_type` is missing.
- For each such assay (with a rich progress bar across the batch):
  - Fetches the description XML from PUG-REST, parses it, and shows the name plus a clickable link to the PubChem assay page.
  - Streams description/protocol text through a pager for easy scrolling (`q` to exit the pager).
  - Prompts for labels with minimal keystrokes:
    - Target Type: `1` target-based, `2` cell-based, `3` other.
    - Bioactivity Type: `1` specific bioactivity, `2` nonspecific bioactivity, `3` other.
    - `s` skips the assay without changes; pressing Enter keeps any existing value from Hit Dexter.
- Writes the chosen labels into the metadata DB and computes `assay_format` accordingly, leaving all other metadata untouched.

---

## Example end‑to‑end for a single AID

Below is a concrete sequence for one assay (e.g. AID 588334):

```bash
cd minimal

# 1. Download aggregated assay table
assay-etl download-assays --aid 588334

# 2. Build or refresh metadata
assay-etl summarize-assays

# 3. Interactively select a column for this assay
assay-etl select-column --aid 588334

# (Optional) Manually annotate target/bioactivity labels using the description pager
assay-etl annotate-metadata --aid 588334

# (Optionally repeat select-column / annotate-metadata for other AIDs.)

# 4. Compute r-scores for all assays with selected columns
assay-etl compute-rscores
```

After this, you will have:

- `outputs/assay_metadata.csv` – assay‑level metadata, including your selections and stats.
- `outputs/assay_rscores.parquet` – combined per‑compound r‑scores (`assay_id, compound_id, smiles, r_score`).
