# Project Structure

This repository is organized so source code stays clean and generated artifacts are isolated.

## Source Code

- `env/`: task environments (Taxi, Frozen Lake)
- `reward_machines/`: TRM/RM wrappers and product dynamics
- `rl_algos/`: baseline and HRM learning code
- `configs/`: experiment configs
- `scripts/`: helper launch scripts
- `run_trm_compare.py`: main launcher for the 4-behavior comparison
- `plot_trm_compare.py`: aggregation/plot generation
- `run_compare_p100.sbatch`: Slurm launcher for the cluster

## Documentation

- `README.md`: original TRM repo usage
- `README_TRM_HRM.md`: digital-time TRM/HRM/CI comparison notes
- `docs/`: extra docs (this file, reports, etc.)

## Generated Artifacts (not for git)

- `artifacts/legacy/Logs_paper/`: legacy TensorBoard logs from paper runs
- `artifacts/legacy/q_learning_log_avg/`: legacy averaged q-learning logs
- `artifacts/dev_runs/`: smoke/debug experiment outputs
- `artifacts/results_main_non_git/`: heavy files moved out of publishable results (for example checkpoints)
- `artifacts/slurm_logs/`: `.out` / `.err` cluster logs
- `**/tensorboard/`, `**/checkpoints/`, `*.out`, `*.err`

## Versioned Results (for GitHub)

- `results_main/`: curated experiment outputs to publish
  - plots
  - per-seed CSV/JSON metrics and summaries

The `artifacts/` section is ignored by `.gitignore`; `results_main/` is intentionally kept versionable.
