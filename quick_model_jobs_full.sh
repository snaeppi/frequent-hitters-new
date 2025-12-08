#!/usr/bin/env bash

set -euo pipefail

# Quick end-to-end sanity check for model-jobs on already-generated datasets.
# Trains regression and threshold models locally (non-UGE) for 3 epochs.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_OUT=${PIPELINE_OUT:-"${ROOT_DIR}/data/processed50"}
SEED=${SEED:-1}
JOBS_DIR=${JOBS_DIR:-"${ROOT_DIR}/integration_artifacts/full_model_jobs"}

echo "[INFO] Using ROOT_DIR=${ROOT_DIR}"
echo "[INFO] Looking for datasets under ${PIPELINE_OUT}"

choose_format() {
  for fmt in biochemical cellular; do
    local reg_path="${PIPELINE_OUT}/${fmt}/${fmt}_regression.parquet"
    if [[ -f "${reg_path}" ]]; then
      echo "${fmt}"
      return 0
    fi
  done
  return 1
}

ASSAY_FORMAT=$(choose_format) || {
  echo "[ERROR] Could not find regression dataset under ${PIPELINE_OUT} (checked biochemical & cellular)." >&2
  exit 1
}

ASSAY_DIR="${PIPELINE_OUT}/${ASSAY_FORMAT}"
REG_PATH="${ASSAY_DIR}/${ASSAY_FORMAT}_regression.parquet"
THRESH_PATH="${ASSAY_DIR}/${ASSAY_FORMAT}_thresholds.json"

for path in "${REG_PATH}" "${THRESH_PATH}"; do
  [[ -f "${path}" ]] || { echo "[ERROR] Missing required dataset: ${path}" >&2; exit 1; }
done

mkdir -p "${JOBS_DIR}/scripts" "${JOBS_DIR}/models" "${JOBS_DIR}/temp"

JOBS_CONFIG="${JOBS_DIR}/fh_jobs_full.yaml"
cat > "${JOBS_CONFIG}" <<EOF
global:
  models_dir: ${JOBS_DIR}/models
  temp_dir: ${JOBS_DIR}/temp
  conda_commands: []
  conda_activate: ""
  cpus: 2
  seed: ${SEED}
  submit: false
  datasets:
    reg: ${REG_PATH}
    thresholds: ${THRESH_PATH}

tasks:
  - type: regression
    job_name: quick_reg_${ASSAY_FORMAT}
    trainval_path: reg
    split_seed: ${SEED}
    target_column: score_seed${SEED}
    compound_min_screens: 50
    compound_screens_column: screens
    screens_weight_mode: linear
    predict_test: true
    epochs: 3
    ensemble_size: 1

  - type: threshold
    job_name: quick_thr_${ASSAY_FORMAT}
    trainval_path: reg
    split_seed: ${SEED}
    metric_column: score_seed${SEED}
    thresholds_json: thresholds
    screens_weight_mode: linear
    predict_test: true
    epochs: 3
    ensemble_size: 1
    thresholds:
      - suffix: p50_95
        lower_percentile: 50
        upper_percentile: 95
EOF

echo "[INFO] Generating job scripts with model-jobs (assay_format=${ASSAY_FORMAT}, seed=${SEED})"

pushd "${ROOT_DIR}/model-jobs" >/dev/null
PYTHONPATH=src python -m model_jobs.cli submit-jobs \
  --config "${JOBS_CONFIG}" \
  --output-dir "${JOBS_DIR}/scripts" \
  --dry-run
popd >/dev/null

echo "[INFO] Running training scripts"
for job in quick_reg_${ASSAY_FORMAT} quick_thr_${ASSAY_FORMAT}_p50_95; do
  script="${JOBS_DIR}/scripts/${job}.sh"
  if [[ -f "${script}" ]]; then
    echo "[INFO] Executing ${script}"
    bash "${script}"
  else
    echo "[WARN] Script ${script} not found; skipping."
  fi
done

echo "[INFO] Completed quick model-jobs run. Outputs under ${JOBS_DIR}/models"
