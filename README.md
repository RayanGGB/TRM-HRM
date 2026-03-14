# TRM-HRM: Digital-Time Timed Reward Machine Experiments

This repository contains:
- the original Timed-RM codebase,
- a **digital-time** comparison pipeline for 4 behaviors:
1. `trm_vanilla`
2. `trm_ci` (TRM + local digital CI/CRM)
3. `trm_hrm`
4. `trm_hrm_ci` (TRM+HRM + CI)

The comparison is designed for reproducibility (fixed seeds, structured outputs, checkpoint/resume, aggregated plots).

## What Is Compared

Main entry point for the new experiments:
- `run_trm_compare.py`

Methods:
- `trm_vanilla`: baseline digital TRM tabular Q-learning.
- `trm_ci`: baseline + local counterfactual imagination (digital-time CRM).
- `trm_hrm`: hierarchical controller over TRM product states.
- `trm_hrm_ci`: HRM + the same local CI mechanism.

Supported digital tasks:
- Taxi (`env/Taxi/disc_vs_cont.txt`)
- Frozen Lake (`env/Frozen_Lake/disc_vs_cont.txt`)

## Quick Start

### 1) Environment setup

```bash
conda create -n TRM_HRL python=3.10 -y
conda activate TRM_HRL
pip install -r requirements.txt
```

Note: `requirements_trm_hrm.txt` currently forwards to `requirements.txt`.

### 2) Run all 4 behaviors locally

```bash
python run_trm_compare.py \
  --alg all4 \
  --env-type both \
  --seeds 42,43,44,45,46,47,48,49,50,51 \
  --total-timesteps 300000 \
  --results-root results_main
```

### 3) Resume interrupted runs

```bash
python run_trm_compare.py \
  --alg all4 \
  --env-type both \
  --seeds 42,43,44,45,46,47,48,49,50,51 \
  --total-timesteps 300000 \
  --results-root results_main \
  --resume
```

### 4) Generate comparison plots

```bash
python plot_trm_compare.py \
  --results-root results_main \
  --env-type both \
  --source eval \
  --metrics episode_discounted_reward,episode_time
```

Plots are written to:
- `results_main/plots/taxi/`
- `results_main/plots/frozen_lake/`

## Slurm (P100) Execution

Use the provided batch file:
- `run_compare_p100.sbatch`

Submit:
```bash
sbatch run_compare_p100.sbatch
```

Resume mode:
```bash
RESUME=1 sbatch run_compare_p100.sbatch
```

Override defaults if needed:
```bash
SEEDS=42,43,44 TOTAL_TIMESTEPS=500000 RESULTS_ROOT=results_main sbatch run_compare_p100.sbatch
```

Slurm logs:
- `artifacts/slurm_logs/<jobname>-<jobid>.out`
- `artifacts/slurm_logs/<jobname>-<jobid>.err`

## Original Timed-RM Baseline Entry

The original script is still available:

```bash
python main.py -e taxi -t env/Taxi/disc_vs_cont.txt -c 1 -m digital -d 1.0 -tt 300000 -n 1
```

Use `run_trm_compare.py` for the new standardized comparison workflow.

## Repository Structure

- `run_trm_compare.py`: training launcher (4 methods + checkpoint/resume).
- `plot_trm_compare.py`: aggregation and plotting.
- `rl_algos/trm_baseline.py`: baseline trainer.
- `rl_algos/trm_hrm.py`: HRM trainer.
- `reward_machines/`: TRM environment/product dynamics.
- `results_main/`: publishable metrics/plots.
- `artifacts/`: non-versioned heavy artifacts and logs.
- `docs/PROJECT_STRUCTURE.md`: structure notes.
- `README_TRM_HRM.md`: implementation details.

## Results and GitHub Hygiene

Expected run folder pattern:
- `results_main/<algorithm>/<task>/seed_<N>/`

Typical files:
- `config.json`
- `summary.json`
- `train_metrics.csv`, `eval_metrics.csv`
- `train_metrics.jsonl`, `eval_metrics.jsonl`

Heavy artifacts are excluded from Git:
- checkpoints (`.pkl`)
- tensorboard event directories
- slurm `.out/.err`

Optional cleanup helper before push:
```bash
python scripts/prepare_results_for_github.py
```

## Legacy Paper Logs

Legacy logs from previous paper experiments are stored in:
- `artifacts/legacy/Logs_paper/`

You can still extract figures from those logs, for example:
```bash
python tb_extract.py -b artifacts/legacy/Logs_paper/disc_vs_cont_Taxi
```
