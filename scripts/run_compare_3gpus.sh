#!/usr/bin/env bash
set -euo pipefail

# Tabular methods are CPU-based; this script uses GPU slots as worker labels.
SEEDS_GPU0="${SEEDS_GPU0:-42,45,48,51}"
SEEDS_GPU1="${SEEDS_GPU1:-43,46,49}"
SEEDS_GPU2="${SEEDS_GPU2:-44,47,50}"

CUDA_VISIBLE_DEVICES=0 python run_trm_compare.py \
  --alg all4 \
  --env-type both \
  --device cuda:0 \
  --seeds "${SEEDS_GPU0}" \
  "$@" &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python run_trm_compare.py \
  --alg all4 \
  --env-type both \
  --device cuda:1 \
  --seeds "${SEEDS_GPU1}" \
  "$@" &
PID1=$!

CUDA_VISIBLE_DEVICES=2 python run_trm_compare.py \
  --alg all4 \
  --env-type both \
  --device cuda:2 \
  --seeds "${SEEDS_GPU2}" \
  "$@" &
PID2=$!

wait "${PID0}" "${PID1}" "${PID2}"
