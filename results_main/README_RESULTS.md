# Published Results

This folder is intended to be pushed to GitHub.

It contains:
- aggregated plots (`plots/`)
- per-run metrics (`train_metrics.csv`, `eval_metrics.csv`, `.jsonl`)
- run configs and summaries (`config.json`, `summary.json`, `run_manifest.json`)

To keep repository size reasonable, heavy artifacts are not kept here:
- checkpoints (`.pkl`)
- tensorboard event folders

Those heavy files are stored in:
- `artifacts/results_main_non_git/`
