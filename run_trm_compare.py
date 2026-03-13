import argparse
import shutil
from pathlib import Path

import gymnasium as gym

from rl_algos.trm_baseline import train_trm_baseline
from rl_algos.trm_experiment_utils import load_json, run_completed, save_json
from rl_algos.trm_hrm import train_trm_hrm
from reward_machines.trm_environment_gym import (
    TimedRewardMachineEnvGym,
    TimedRewardMachineWrapperGym,
)


DEFAULT_TRM_PATHS = {
    "taxi": "env/Taxi/disc_vs_cont.txt",
    "frozen_lake": "env/Frozen_Lake/disc_vs_cont.txt",
}

ENV_IDS = {
    "taxi": "CustomTaxi-v0",
    "frozen_lake": "FrozenLakeEnv",
}


def _register_env_if_needed(env_type):
    env_id = ENV_IDS[env_type]
    if env_id in gym.registry:
        return env_id

    if env_type == "taxi":
        gym.envs.registration.register(
            id=env_id,
            entry_point="env.Taxi.taxi:TaxiEnv",
            kwargs={"render_mode": "rgb_array"},
            max_episode_steps=100,
        )
    elif env_type == "frozen_lake":
        gym.envs.registration.register(
            id=env_id,
            entry_point="env.Frozen_Lake.frozen_lake:FrozenLakeEnv",
            kwargs={
                "map_name": "8x8_3Goals",
                "is_slippery": True,
                "success_rate": 0.8,
                "reward_schedule": (0, 0, 0),
            },
            max_episode_steps=100,
        )
    else:
        raise ValueError(f"Unsupported env_type: {env_type}")
    return env_id


def _make_wrapped_env(
    env_type,
    trm_name,
    add_ci,
    gamma,
    discretization_param,
    seed,
    crm_nums,
    crm_option,
    crm_radius,
):
    env_id = _register_env_if_needed(env_type)
    base_env = gym.make(env_id, render_mode="rgb_array")
    rm_env = TimedRewardMachineEnvGym(
        env=base_env,
        rm_files=[str(trm_name)],
        gamma=gamma,
        discretization_param=discretization_param,
        seed=seed,
    )
    wrapped = TimedRewardMachineWrapperGym(
        env=rm_env,
        add_crm=add_ci,
        add_rs=False,
        gamma=gamma,
        rs_gamma=gamma,
        crm_nums=crm_nums,
        crm_option=crm_option,
        crm_radius=crm_radius,
    )
    return wrapped


def _parse_seeds(args):
    if args.seeds:
        return [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    return [args.seed_start + i for i in range(args.num_seeds)]


def _parse_tasks(args):
    if args.env_type == "both":
        envs = ["taxi", "frozen_lake"]
    else:
        envs = [args.env_type]

    tasks = []
    for env_type in envs:
        trm_name = args.trm_name if args.trm_name else DEFAULT_TRM_PATHS[env_type]
        tasks.append((env_type, trm_name))
    return tasks


def _get_methods(args):
    if args.alg == "all4":
        return [
            {"name": "trm_vanilla", "trainer": "trm_baseline", "use_ci": False},
            {"name": "trm_ci", "trainer": "trm_baseline", "use_ci": True},
            {"name": "trm_hrm", "trainer": "trm_hrm", "use_ci": False},
            {"name": "trm_hrm_ci", "trainer": "trm_hrm", "use_ci": True},
        ]
    if args.alg == "both":
        return [
            {"name": "trm_baseline", "trainer": "trm_baseline", "use_ci": bool(args.add_ci)},
            {"name": "trm_hrm", "trainer": "trm_hrm", "use_ci": bool(args.add_ci)},
        ]
    if args.alg == "baseline":
        return [{"name": "trm_baseline", "trainer": "trm_baseline", "use_ci": bool(args.add_ci)}]
    if args.alg == "trm_hrm":
        return [{"name": "trm_hrm", "trainer": "trm_hrm", "use_ci": bool(args.add_ci)}]
    raise ValueError(args.alg)


def _build_common_config(args, env_type, trm_name, seed, use_ci):
    return {
        "env_type": env_type,
        "trm_name": trm_name,
        "seed": seed,
        "gamma": args.gamma,
        "epsilon": args.epsilon,
        "epsilon_decay": args.epsilon_decay,
        "lr": args.lr,
        "lr_decay": args.lr_decay,
        "total_timesteps": args.total_timesteps,
        "q_init": args.q_init,
        "min_epsilon": args.min_epsilon,
        "learning_starts": args.learning_starts,
        "log_every_steps": args.log_every_steps,
        "eval_every_steps": args.eval_every_steps,
        "use_crm": bool(use_ci),
        "use_rs": False,
        "discretization_param": args.discretization_param,
        "checkpoint_every_steps": args.checkpoint_every_steps,
        "checkpoint_every_episodes": args.checkpoint_every_episodes,
        "checkpoint_every_minutes": args.checkpoint_every_minutes,
        "save_milestones": args.save_milestones,
        "device": args.device,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run digital-time TRM comparisons (vanilla/CI/HRM/HRM+CI) with checkpoint/resume support."
    )
    parser.add_argument("--alg", choices=["baseline", "trm_hrm", "both", "all4"], default="all4")
    parser.add_argument("--env-type", choices=["taxi", "frozen_lake", "both"], default="both")
    parser.add_argument("--trm-name", default=None, help="Override TRM path (single env_type only).")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds, e.g. 42,43,44")
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--num-seeds", type=int, default=10)

    parser.add_argument("--total-timesteps", type=int, default=300000)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--epsilon", type=float, default=0.9)
    parser.add_argument("--epsilon-decay", type=float, default=0.999)
    parser.add_argument("--min-epsilon", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.8)
    parser.add_argument("--lr-decay", type=float, default=0.999)
    parser.add_argument("--q-init", type=float, default=10.0)
    parser.add_argument("--learning-starts", type=int, default=0)
    parser.add_argument("--log-every-steps", type=int, default=100)
    parser.add_argument("--eval-every-steps", type=int, default=2000)

    parser.add_argument("--add-ci", type=int, choices=[0, 1], default=1)
    parser.add_argument("--crm-nums", type=int, default=15)
    parser.add_argument("--crm-option", type=int, default=3)
    parser.add_argument("--crm-radius", type=int, default=1)
    parser.add_argument("--discretization-param", type=float, default=1.0)

    parser.add_argument("--hrm-lr", type=float, default=0.8)
    parser.add_argument("--hrm-lr-decay", type=float, default=0.999)
    parser.add_argument("--r-plus", type=float, default=1.0)
    parser.add_argument("--r-minus", type=float, default=0.0)
    parser.add_argument("--parallel-option-updates", action="store_true", default=True)
    parser.add_argument("--no-parallel-option-updates", dest="parallel_option_updates", action="store_false")

    parser.add_argument("--checkpoint-every-steps", type=int, default=20000)
    parser.add_argument("--checkpoint-every-episodes", type=int, default=25)
    parser.add_argument("--checkpoint-every-minutes", type=float, default=15.0)
    parser.add_argument("--save-milestones", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.env_type == "both" and args.trm_name is not None:
        raise ValueError("--trm-name override is only supported when --env-type is not 'both'.")

    if args.device != "cpu":
        print(
            f"[Info] device={args.device} requested. These methods are tabular and CPU-based; "
            "device is kept only for launcher compatibility."
        )

    seeds = _parse_seeds(args)
    tasks = _parse_tasks(args)
    methods = _get_methods(args)
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    run_manifest = {
        "algorithms": [m["name"] for m in methods],
        "tasks": tasks,
        "seeds": seeds,
        "args": vars(args),
        "runs": [],
    }

    for env_type, trm_name in tasks:
        for seed in seeds:
            for method in methods:
                algorithm = method["name"]
                trainer = method["trainer"]
                use_ci = bool(method["use_ci"])
                run_dir = results_root / algorithm / env_type / f"seed_{seed}"
                summary_path = run_dir / "summary.json"
                latest_ckpt = run_dir / "checkpoints" / "latest.pkl"

                if run_dir.exists() and args.force:
                    shutil.rmtree(run_dir)

                if run_dir.exists() and not args.force:
                    summary = load_json(summary_path, default={})
                    if run_completed(summary_path):
                        run_manifest["runs"].append(
                            {
                                "algorithm": algorithm,
                                "trainer": trainer,
                                "use_ci": use_ci,
                                "env_type": env_type,
                                "seed": seed,
                                "status": "skipped_completed",
                            }
                        )
                        print(f"[Skip] {algorithm} {env_type} seed={seed}: completed run exists.")
                        continue
                    if not args.resume:
                        run_manifest["runs"].append(
                            {
                                "algorithm": algorithm,
                                "trainer": trainer,
                                "use_ci": use_ci,
                                "env_type": env_type,
                                "seed": seed,
                                "status": "skipped_incomplete_use_resume_or_force",
                            }
                        )
                        print(
                            f"[Skip] {algorithm} {env_type} seed={seed}: incomplete run exists. "
                            "Use --resume to continue or --force to restart."
                        )
                        continue
                    if args.resume and not latest_ckpt.exists():
                        print(
                            f"[Warn] {algorithm} {env_type} seed={seed}: --resume set but no checkpoint found. "
                            "Starting from scratch."
                        )

                run_dir.mkdir(parents=True, exist_ok=True)
                common = _build_common_config(args, env_type, trm_name, seed, use_ci)

                train_env = _make_wrapped_env(
                    env_type=env_type,
                    trm_name=trm_name,
                    add_ci=use_ci,
                    gamma=args.gamma,
                    discretization_param=args.discretization_param,
                    seed=seed,
                    crm_nums=args.crm_nums,
                    crm_option=args.crm_option,
                    crm_radius=args.crm_radius,
                )
                eval_env = _make_wrapped_env(
                    env_type=env_type,
                    trm_name=trm_name,
                    add_ci=use_ci,
                    gamma=args.gamma,
                    discretization_param=args.discretization_param,
                    seed=seed + 99999,
                    crm_nums=args.crm_nums,
                    crm_option=args.crm_option,
                    crm_radius=args.crm_radius,
                )

                try:
                    if trainer == "trm_baseline":
                        config = dict(common)
                        config["algorithm"] = algorithm
                        summary = train_trm_baseline(
                            env=train_env,
                            eval_env=eval_env,
                            run_dir=run_dir,
                            config=config,
                            resume=args.resume,
                        )
                    else:
                        config = dict(common)
                        config.update(
                            {
                                "algorithm": algorithm,
                                "hrm_lr": args.hrm_lr,
                                "hrm_lr_decay": args.hrm_lr_decay,
                                "r_plus": args.r_plus,
                                "r_minus": args.r_minus,
                                "parallel_option_updates": args.parallel_option_updates,
                            }
                        )
                        summary = train_trm_hrm(
                            env=train_env,
                            eval_env=eval_env,
                            run_dir=run_dir,
                            config=config,
                            resume=args.resume,
                        )
                finally:
                    train_env.close()
                    eval_env.close()

                run_manifest["runs"].append(
                    {
                        "algorithm": algorithm,
                        "trainer": trainer,
                        "use_ci": use_ci,
                        "env_type": env_type,
                        "seed": seed,
                        "status": "completed" if summary["completed"] else "interrupted",
                        "summary": summary,
                    }
                )
                print(
                    f"[Done] {algorithm} {env_type} seed={seed} "
                    f"completed={summary['completed']} steps={summary['global_step']}"
                )

    save_json(results_root / "run_manifest.json", run_manifest)


if __name__ == "__main__":
    main()
