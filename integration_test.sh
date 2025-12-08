#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACTS_DIR="${ROOT_DIR}/integration_artifacts"
RAW_DIR="${ARTIFACTS_DIR}/raw"
PIPELINE_OUT="${ARTIFACTS_DIR}/pipeline_outputs"
JOBS_DIR="${ARTIFACTS_DIR}/jobs"

mkdir -p "${RAW_DIR}" "${PIPELINE_OUT}" "${JOBS_DIR}"

echo "[INFO] Using ROOT_DIR=${ROOT_DIR}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found. Activate an environment with the project dependencies before running this script." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"
echo "[INFO] Python in current environment: $(python -V)"

echo "[INFO] Installing local packages into environment (no deps)"
pushd "${ROOT_DIR}/chemprop" >/dev/null
python -m pip install --no-deps .
popd >/dev/null

pushd "${ROOT_DIR}/dataset-pipeline" >/dev/null
python -m pip install --no-deps .
popd >/dev/null

pushd "${ROOT_DIR}/model-jobs" >/dev/null
python -m pip install --no-deps .
popd >/dev/null

pushd "${ROOT_DIR}/assay-cleaning" >/dev/null
python -m pip install --no-deps .
popd >/dev/null

echo "[INFO] Creating small subsets from assay_rscores.parquet"

python << 'PY'
from pathlib import Path

import polars as pl

root = Path(__file__).resolve().parent
hits_path = root / "assay-etl/outputs/assay_rscores.parquet"
if not hits_path.exists():
    raise SystemExit(f"Missing assay_rscores.parquet at {hits_path}")

artifacts = root / "integration_artifacts"
raw_dir = artifacts / "raw"
raw_dir.mkdir(parents=True, exist_ok=True)

print(f"[PY] Sampling hits from {hits_path}")
hits = pl.read_parquet(hits_path)
if hits.height > 200_000:
    hits = hits.sample(200_000, shuffle=True)

subset_path = raw_dir / "assay_rscores_subset.parquet"
print(f"[PY] Writing {hits.height} sampled rows to {subset_path}")
hits.write_parquet(subset_path)

PY

HTS_SUBSET="${RAW_DIR}/assay_rscores_subset.parquet"
ASSAY_META="${ROOT_DIR}/assay-etl/outputs/assay_metadata.csv"

if [[ ! -f "${ASSAY_META}" ]]; then
  echo "[ERROR] Missing assay_metadata.csv at ${ASSAY_META}" >&2
  exit 1
fi

echo "[INFO] Running assay-cleaning CLI on sampled HTS data"

python -m assay_cleaning.cli clean \
  --hts-file "${HTS_SUBSET}" \
  --assay-props-file "${ASSAY_META}" \
  --id-col "compound_id" \
  --smiles-col "smiles" \
  --assay-col "assay_id" \
  --assay-format-col "assay_format" \
  --biochemical-format "biochemical" \
  --cellular-format "cellular" \
  --score-col "r_score" \
  --score-threshold 3.0 \
  --biochemical-out "${RAW_DIR}/biochemical_hits_subset.parquet" \
  --cellular-out "${RAW_DIR}/cellular_hits_subset.parquet" \
  --rename-cols

BIO_HITS="${RAW_DIR}/biochemical_hits_subset.parquet"
CELL_HITS="${RAW_DIR}/cellular_hits_subset.parquet"

if [[ ! -f "${BIO_HITS}" && ! -f "${CELL_HITS}" ]]; then
  echo "[ERROR] No biochemical or cellular subsets were written. Check assay_metadata.csv contents." >&2
  exit 1
fi

echo "[INFO] Running dataset pipeline on subsets"

PIPELINE_CMD=(python -m dataset_pipeline.cli)
PIPELINE_ARGS=(
  assay_format=both
  "paths.output_root=${PIPELINE_OUT}"
  "filters.min_screens_per_assay=1"
  "filters.min_screens_per_compound=1"
  "filters.min_screens_for_prior_fit=1"
)

if [[ -f "${BIO_HITS}" ]]; then
  PIPELINE_ARGS+=("paths.input.biochemical=${BIO_HITS}")
fi
if [[ -f "${CELL_HITS}" ]]; then
  PIPELINE_ARGS+=("paths.input.cellular=${CELL_HITS}")
fi

echo "[INFO] Pipeline command:"
printf '  %q' "${PIPELINE_CMD[@]}" "${PIPELINE_ARGS[@]}"
echo

"${PIPELINE_CMD[@]}" "${PIPELINE_ARGS[@]}"

ASSAY_FORMAT=""
if [[ -f "${PIPELINE_OUT}/biochemical/biochemical_regression.parquet" ]]; then
  ASSAY_FORMAT="biochemical"
elif [[ -f "${PIPELINE_OUT}/cellular/cellular_regression.parquet" ]]; then
  ASSAY_FORMAT="cellular"
else
  echo "[ERROR] No regression outputs found under ${PIPELINE_OUT}" >&2
  exit 1
fi

echo "[INFO] Using assay_format=${ASSAY_FORMAT} for downstream training"

echo "[INFO] Preparing model-jobs config"

JOBS_CONFIG="${JOBS_DIR}/fh_jobs.yaml"

cat > "${JOBS_CONFIG}" <<EOF
global:
  models_dir: ${ARTIFACTS_DIR}/models
  temp_dir: ${ARTIFACTS_DIR}/temp
  conda_commands: []
  conda_activate: ""
  cpus: 2
  seed: 1337
  submit: false
  datasets:
    bio_reg: ${PIPELINE_OUT}/${ASSAY_FORMAT}/${ASSAY_FORMAT}_regression.parquet
    bio_mt: ${PIPELINE_OUT}/${ASSAY_FORMAT}/${ASSAY_FORMAT}_multilabel.parquet
    bio_thresholds: ${PIPELINE_OUT}/${ASSAY_FORMAT}/${ASSAY_FORMAT}_thresholds.json

tasks:
  - type: multilabel
    job_name: it_mt_bio
    trainval_path: bio_mt
    predict_test: true
    split_seed: 1337
    epochs: 3
    ensemble_size: 1

  - type: regression
    job_name: it_reg_bio
    trainval_path: bio_reg
    split_seed: 1337
    target_column: score
    compound_min_screens: 1
    compound_screens_column: screens
    predict_test: true
    epochs: 3

  - type: threshold
    job_name: it_thr_bio
    trainval_path: bio_reg
    split_seed: 1337
    thresholds_json: bio_thresholds
    metric_column: score
    target_column: target
    compound_min_screens: 1
    compound_screens_column: screens
    predict_test: true
    thresholds:
      - suffix: p50_95
        lower_percentile: 50
        upper_percentile: 95
    epochs: 3
EOF

echo "[INFO] Generating job scripts with model-jobs"

pushd "${ROOT_DIR}/model-jobs" >/dev/null
PYTHONPATH=src python -m model_jobs.cli submit-jobs \
  --config "${JOBS_CONFIG}" \
  --output-dir "${JOBS_DIR}/scripts" \
  --dry-run
popd >/dev/null

echo "[INFO] Running selected training + prediction scripts"

# Execute a representative regression job to keep runtime minimal.
for job in it_mt_bio it_reg_bio it_thr_bio_p50_95; do
  script="${JOBS_DIR}/scripts/${job}.sh"
  if [[ -f "${script}" ]]; then
    echo "[INFO] Executing ${script}"
    bash "${script}"
  else
    echo "[WARN] Script ${script} not found; skipping."
  fi
done

echo "[INFO] Checking for expected outputs"

test -f "${ARTIFACTS_DIR}/models/it_reg_bio/predictions/test_preds.csv" || {
  echo "[ERROR] Missing test predictions for it_reg_bio" >&2
  exit 1
}

echo "[INFO] Integration test completed successfully."
