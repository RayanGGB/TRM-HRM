#!/bin/bash
#SBATCH --job-name=trm_compare
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=23:50:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

# Conda (important en batch non-interactif)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate TRM_HRL

echo "ENV: $CONDA_DEFAULT_ENV"
which python
python --version

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# Parametres experiment
RESULTS_ROOT=${RESULTS_ROOT:-results_main}
SEEDS=${SEEDS:-42,43,44,45,46,47,48,49,50,51}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-300000}
RESUME=${RESUME:-0}

CMD=(
  python run_trm_compare.py
  --alg all4
  --env-type both
  --seeds "$SEEDS"
  --total-timesteps "$TOTAL_TIMESTEPS"
  --results-root "$RESULTS_ROOT"
)

if [ "$RESUME" = "1" ]; then
  CMD+=(--resume)
fi

echo "Running: ${CMD[*]}"
srun "${CMD[@]}"
