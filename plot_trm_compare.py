import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _collect_metric_frames(results_root, algorithm, env_type, source_file):
    alg_env_root = Path(results_root) / algorithm / env_type
    if not alg_env_root.exists():
        return []
    frames = []
    for seed_dir in sorted(alg_env_root.glob("seed_*")):
        metrics_path = seed_dir / source_file
        if not metrics_path.exists():
            continue
        df = pd.read_csv(metrics_path)
        if "global_step" not in df.columns:
            continue
        df = df.sort_values("global_step").copy()
        df["seed"] = seed_dir.name
        frames.append(df)
    return frames


def _aggregate(frames, metric_col):
    if not frames:
        return None
    all_steps = sorted({int(s) for df in frames for s in df["global_step"].tolist()})
    if not all_steps:
        return None

    values = []
    for df in frames:
        if metric_col not in df.columns:
            continue
        series = df.set_index("global_step")[metric_col].reindex(all_steps)
        values.append(series.to_numpy(dtype=float))

    if not values:
        return None

    arr = np.vstack(values)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    count = np.sum(~np.isnan(arr), axis=0)
    return pd.DataFrame(
        {
            "global_step": all_steps,
            "mean": mean,
            "std": std,
            "num_seeds": count,
        }
    )


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate and plot TRM comparison metrics."
    )
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--env-type", choices=["taxi", "frozen_lake", "both"], default="both")
    parser.add_argument("--source", choices=["eval", "train"], default="eval")
    parser.add_argument(
        "--metrics",
        default="episode_discounted_reward,episode_time",
        help="Comma-separated metric columns to plot.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--algorithms",
        default="trm_vanilla,trm_ci,trm_hrm,trm_hrm_ci",
    )
    args = parser.parse_args()

    env_types = ["taxi", "frozen_lake"] if args.env_type == "both" else [args.env_type]
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    algorithms = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    source_file = "eval_metrics.csv" if args.source == "eval" else "train_metrics.csv"

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.results_root) / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    for env_type in env_types:
        env_out = output_dir / env_type
        env_out.mkdir(parents=True, exist_ok=True)
        for metric in metrics:
            plt.figure(figsize=(8, 5), dpi=140)
            plotted = False
            for algorithm in algorithms:
                frames = _collect_metric_frames(args.results_root, algorithm, env_type, source_file)
                agg = _aggregate(frames, metric)
                if agg is None or agg.empty:
                    continue
                x = agg["global_step"].to_numpy(dtype=float)
                y = agg["mean"].to_numpy(dtype=float)
                s = agg["std"].to_numpy(dtype=float)
                plt.plot(x, y, label=algorithm, linewidth=1.8)
                if np.nanmax(s) > 0:
                    plt.fill_between(x, y - s, y + s, alpha=0.18)
                agg.to_csv(env_out / f"{algorithm}_{args.source}_{metric}.csv", index=False)
                plotted = True

            if not plotted:
                plt.close()
                continue
            plt.title(f"{env_type}: {metric} ({args.source})")
            plt.xlabel("Global step")
            plt.ylabel(metric)
            plt.grid(True, alpha=0.3)
            plt.legend(frameon=False)
            plt.tight_layout()
            plot_path = env_out / f"{args.source}_{metric}.png"
            plt.savefig(plot_path)
            plt.close()
            print(f"[Plot] {plot_path}")


if __name__ == "__main__":
    main()
