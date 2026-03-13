# TRM Digital-Time Comparison (4 Behaviors)

This adds a **digital-time only** experimental comparison in `Timed-RM` with 4 behaviors:
- `trm_vanilla`: original TRM digital tabular baseline (no CI).
- `trm_ci`: TRM baseline + digital-time Counterfactual Imagining (CI/CRM).
- `trm_hrm`: hierarchical TRM+HRM (no CI).
- `trm_hrm_ci`: hierarchical TRM+HRM + digital-time CI.

The implementation reuses:
- TRM repo environment/product dynamics (`reward_machines/trm_environment_gym.py`, `timed_reward_machine.py`).
- TRM baseline action/value logic (`rl_algos/q_learn_ct.py` helpers).
- RM HRM structure (option-controller pattern) from `reward_machines-master/reward_machines/rl_agents/hrm/hrm.py` and `reward_machines/rm_environment.py`.

## Added Files

- `run_trm_compare.py`: main launcher for baseline/HRM comparison.
- `rl_algos/trm_baseline.py`: checkpointable baseline trainer.
- `rl_algos/trm_hrm.py`: checkpointable TRM+HRM trainer.
- `rl_algos/trm_experiment_utils.py`: checkpoint, metrics, signal handling, parsing utils.
- `plot_trm_compare.py`: aggregation and plotting from run folders.
- `configs/trm_digital_compare_default.json`: default config snapshot.
- `scripts/run_all_baselines.sh`
- `scripts/run_all_trm_hrm.sh`
- `scripts/run_all_4behaviors.sh`
- `scripts/run_compare_3gpus.sh`

## Dependencies

Use the `Timed-RM` dependencies:

```bash
cd Timed-RM
pip install -r requirements.txt
```

`reward_machines-master` is used as a local reference for HRM design; this new comparison code runs from `Timed-RM`.

## Digital Baseline Entry Point (Original Repo)

Original digital-time TRM baseline entry point:

```bash
python main.py -e taxi -t env/Taxi/disc_vs_cont.txt -c 1 -m digital -d 1.0 -tt 300000 -n 1
```

In this extension, baseline comparison runs through `run_trm_compare.py` to add checkpoint/resume and standardized outputs while preserving baseline update logic.

## Run Commands

### Fresh baseline run

```bash
python run_trm_compare.py --alg baseline --env-type taxi --seeds 42 --total-timesteps 300000
```

### Resume baseline run

```bash
python run_trm_compare.py --alg baseline --env-type taxi --seeds 42 --total-timesteps 300000 --resume
```

### Fresh TRM+HRM run

```bash
python run_trm_compare.py --alg trm_hrm --env-type taxi --seeds 42 --total-timesteps 300000
```

### Resume TRM+HRM run

```bash
python run_trm_compare.py --alg trm_hrm --env-type taxi --seeds 42 --total-timesteps 300000 --resume
```

### Full 4-behavior comparison on paper digital tasks (Taxi + Frozen Lake)

```bash
python run_trm_compare.py --alg all4 --env-type both --seeds 42,43,44,45,46,47,48,49,50,51 --total-timesteps 300000
```

## Checkpoint / Resume Design

Each run writes:

```text
results/<algorithm>/<task>/seed_<N>/
  config.json
  summary.json
  train_metrics.csv
  train_metrics.jsonl
  eval_metrics.csv
  eval_metrics.jsonl
  tensorboard/
  checkpoints/
    latest.pkl
    final.pkl
    step_XXXXXXXXX.pkl   (if milestone saving enabled)
```

Checkpoint payload contains:
- Q tables (baseline or HRM controller/options),
- exploration state (`epsilon_*`),
- learning rates,
- global step / episode counters,
- RNG states (Python / NumPy / Torch when available),
- current metric accumulators,
- active option state (HRM),
- run config,
- placeholders for optimizer/replay buffer (`None` for tabular methods).

Periodic saving:
- `--checkpoint-every-steps`
- `--checkpoint-every-episodes`
- `--checkpoint-every-minutes`
- signal-triggered checkpoint on `SIGINT`/`SIGTERM`.

## Partial Completion and Re-runs

Default behavior:
- completed seeds are skipped automatically,
- incomplete existing runs are skipped unless `--resume` is passed.

To restart from scratch:

```bash
python run_trm_compare.py ... --force
```

## Result Aggregation and Plotting

Generate comparison plots:

```bash
python plot_trm_compare.py --results-root results --env-type both --source eval --metrics episode_discounted_reward,episode_time
```

Outputs:
- `results/plots/<task>/eval_episode_discounted_reward.png`
- `results/plots/<task>/eval_episode_time.png`
- per-algorithm aggregated CSVs in the same folder.

The plotting script tolerates missing seeds and uses available runs only.

## 3-GPU / 3-Worker Strategy

These methods are tabular and effectively CPU-bound. Use 3 GPUs as 3 worker slots:

```bash
bash scripts/run_compare_3gpus.sh
```

or manually:

```bash
CUDA_VISIBLE_DEVICES=0 python run_trm_compare.py --alg all4 --env-type both --seeds 42,45,48,51 --device cuda:0 &
CUDA_VISIBLE_DEVICES=1 python run_trm_compare.py --alg all4 --env-type both --seeds 43,46,49 --device cuda:1 &
CUDA_VISIBLE_DEVICES=2 python run_trm_compare.py --alg all4 --env-type both --seeds 44,47,50 --device cuda:2 &
wait
```

## Notes on TRM+HRM Model Choices

- Options are indexed by target TRM state `u'` for each source `u`.
- Self-loop options are excluded.
- Low-level actions are augmented `(delay, env_action)`.
- Option termination: TRM state changes (`u_next != u_source`) or episode ends.
- Local option reward:
  - base TRM product reward,
  - `+r_plus` when hitting target `u'`,
  - `+r_minus` when leaving `u` via wrong target.
- Low-level TD discount uses elapsed digital time `gamma^(d+1)` (with repo discretization semantics).
- High-level update is semi-MDP and triggered only at option termination.
- Parallel off-policy low-level updates for all options from the current source state are enabled by default (`--parallel-option-updates`).
- CI (digital-time CRM) is local: for each real transition it keeps `(s, s', a, u)` fixed, varies only clock valuation `v` within `--crm-radius` and delays `d` that satisfy relevant outgoing guards for the observed label, then keeps top `--crm-nums` transitions by reward.
