#!/usr/bin/env bash
set -euo pipefail

SEEDS="${SEEDS:-42,43,44,45,46,47,48,49,50,51}"
RESULTS_ROOT="${RESULTS_ROOT:-results_main}"

python run_trm_compare.py \
  --alg all4 \
  --env-type both \
  --seeds "${SEEDS}" \
  --results-root "${RESULTS_ROOT}" \
  "$@"
