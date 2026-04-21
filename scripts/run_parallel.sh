#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-4}"
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "$LOG_DIR" results models

ENVS=(
  Ant-v4
  HalfCheetah-v4
  Hopper-v4
  Walker2d-v4
  InvertedDoublePendulum-v4
)

mapfile -t SEEDS < <("$PYTHON_BIN" - <<'PY'
import random

for seed in random.sample(range(100000), 5):
  print(seed)
PY
)

BATCH_ID="$(date +%Y%m%d_%H%M%S)"
SEED_SUMMARY_FILE="$LOG_DIR/seeds_${BATCH_ID}.txt"

{
  echo "batch_id=${BATCH_ID}"
  echo "python_bin=${PYTHON_BIN}"
  echo "max_parallel_jobs=${MAX_PARALLEL_JOBS}"
  echo "log_dir=${LOG_DIR}"
  echo "seeds:"
  printf '%s\n' "${SEEDS[@]}"
} > "$SEED_SUMMARY_FILE"

echo "[seed-list] ${SEED_SUMMARY_FILE}"

CASES=(
  "PT_PLIF|Yes|PLIF"
)

GPUS=(0 1)
gpu_index=0

wait_for_slot() {
  while [[ "$(jobs -pr | wc -l)" -ge "$MAX_PARALLEL_JOBS" ]]; do
    wait -n
  done
}

run_one() {
  local env_name="$1"
  local proxy_flag="$2"
  local neuron_name="$3"
  local seed_value="$4"
  local label="$5"
  local gpu_id="$6"

  local log_file="$LOG_DIR/${label}_${env_name}_seed${seed_value}_gpu${gpu_id}.log"

  echo "[launch] ${label} env=${env_name} seed=${seed_value} gpu=${gpu_id} log=${log_file}"
  CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" main.py \
    --env "$env_name" \
    --proxy "$proxy_flag" \
    --spiking_neurons "$neuron_name" \
    --seed "$seed_value" \
    >"$log_file" 2>&1
}

for case in "${CASES[@]}"; do
  IFS='|' read -r label proxy_flag neuron_name <<<"$case"

  for env_name in "${ENVS[@]}"; do
    for seed_value in "${SEEDS[@]}"; do
      wait_for_slot
      gpu_id="${GPUS[$((gpu_index % ${#GPUS[@]}))]}"
      gpu_index=$((gpu_index + 1))
      run_one "$env_name" "$proxy_flag" "$neuron_name" "$seed_value" "$label" "$gpu_id" &
    done
  done
done

wait
