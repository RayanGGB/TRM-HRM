import random
import time
from pathlib import Path

import numpy as np

from avg_tb import SummaryWriter
from rl_algos.q_learn_ct import (
    get_best_delay_action,
    get_qmax,
    init_Q_delay_action,
)
from rl_algos.trm_experiment_utils import (
    CheckpointManager,
    GracefulTerminator,
    MetricsLogger,
    capture_rng_states,
    discounted_time_from_action,
    restore_rng_states,
    save_json,
    set_global_seed,
    valid_augmented_actions,
)


def _evaluate_baseline_policy(env, Q, q_init, gamma, eval_seed, max_eval_steps=2000):
    delays = env.env.delay_space
    env_actions = env.env.env_action_space
    max_constant_array = np.asarray(list(env.env.max_constant_dict.values()), dtype=float)
    num_clocks = env.env.num_clocks
    discretization_param = env.env.discretization_param
    actions = env.env.action_combinations

    obs, _ = env.reset(seed=eval_seed, options={"random": False})
    s = tuple(obs)
    init_Q_delay_action(Q, s, delays, env_actions, q_init)
    done = False
    episode_reward = 0.0
    episode_time = 0.0
    episode_discounted_reward = 0.0
    steps = 0

    while not done and steps < max_eval_steps:
        valid_actions = valid_augmented_actions(
            s[-num_clocks:],
            actions,
            max_constant_array,
            discretization_param,
        )
        best_delay, best_action = get_best_delay_action(
            Q, s, delays, env_actions, max_constant_array, discretization_param, num_clocks
        )
        greedy_action = (best_delay, best_action)
        if greedy_action not in valid_actions:
            greedy_action = valid_actions[0]
        action_index = env.env.action_to_index[greedy_action]
        sn, r, term, trunc, _ = env.step(action_index)
        done = term or trunc
        t = discounted_time_from_action(greedy_action, discretization_param)
        episode_reward += r
        episode_discounted_reward += (gamma ** episode_time) * r
        episode_time += t
        s = tuple(sn)
        init_Q_delay_action(Q, s, delays, env_actions, q_init)
        steps += 1

    return {
        "episode_reward": float(episode_reward),
        "episode_time": float(episode_time),
        "episode_discounted_reward": float(episode_discounted_reward),
    }


def train_trm_baseline(
    env,
    eval_env,
    run_dir,
    config,
    resume=False,
):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "config.json", config)

    algorithm_name = str(config.get("algorithm", "trm_baseline"))

    metrics = MetricsLogger(run_dir)
    ckpt = CheckpointManager(
        run_dir=run_dir,
        save_every_steps=config["checkpoint_every_steps"],
        save_every_episodes=config["checkpoint_every_episodes"],
        save_every_seconds=int(config["checkpoint_every_minutes"] * 60),
        save_milestones=config["save_milestones"],
    )
    terminator = GracefulTerminator()

    seed = int(config["seed"])
    set_global_seed(seed)

    delays = env.env.delay_space
    env_actions = env.env.env_action_space
    max_constant_array = np.asarray(list(env.env.max_constant_dict.values()), dtype=float)
    num_clocks = env.env.num_clocks
    actions = env.env.action_combinations
    discretization_param = env.env.discretization_param

    Q = {}
    global_step = 0
    num_episodes = 0
    eval_id = 0

    epsilon_delay = float(config["epsilon"])
    epsilon_action = float(config["epsilon"])
    lr = float(config["lr"])

    running_reward = 0.0
    running_time = 0.0
    running_discounted_reward = 0.0
    current_episode_reward = 0.0
    current_episode_reward_discounted = 0.0
    current_episode_time = 0.0

    episode_reward = 0.0
    episode_time = 0.0
    episode_discounted_reward = 0.0

    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    obs, _ = env.reset(seed=seed, options={"random": False})
    s = tuple(obs)
    init_state = s
    init_Q_delay_action(Q, s, delays, env_actions, config["q_init"])

    if resume:
        state = ckpt.load_latest()
        if state is not None:
            Q = state["Q"]
            global_step = int(state["global_step"])
            num_episodes = int(state["num_episodes"])
            eval_id = int(state["eval_id"])
            epsilon_delay = float(state["epsilon_delay"])
            epsilon_action = float(state["epsilon_action"])
            lr = float(state["lr"])
            running_reward = float(state["running_reward"])
            running_time = float(state["running_time"])
            running_discounted_reward = float(state["running_discounted_reward"])
            current_episode_reward = float(state["current_episode_reward"])
            current_episode_reward_discounted = float(state["current_episode_reward_discounted"])
            current_episode_time = float(state["current_episode_time"])
            episode_reward = float(state["episode_reward"])
            episode_time = float(state["episode_time"])
            episode_discounted_reward = float(state["episode_discounted_reward"])
            restore_rng_states(state.get("rng_state"))
            # We intentionally restart from a clean env episode on resume.
            obs, _ = env.reset(seed=seed, options={"random": False})
            s = tuple(obs)
            init_state = tuple(state.get("init_state", s))
            init_Q_delay_action(Q, s, delays, env_actions, config["q_init"])

    def build_checkpoint_payload(reason):
        return {
            "algorithm": algorithm_name,
            "reason": reason,
            "Q": Q,
            "global_step": global_step,
            "num_episodes": num_episodes,
            "eval_id": eval_id,
            "epsilon_delay": epsilon_delay,
            "epsilon_action": epsilon_action,
            "lr": lr,
            "running_reward": running_reward,
            "running_time": running_time,
            "running_discounted_reward": running_discounted_reward,
            "current_episode_reward": current_episode_reward,
            "current_episode_reward_discounted": current_episode_reward_discounted,
            "current_episode_time": current_episode_time,
            "episode_reward": episode_reward,
            "episode_time": episode_time,
            "episode_discounted_reward": episode_discounted_reward,
            "init_state": init_state,
            "rng_state": capture_rng_states(),
            "optimizer_state": None,
            "replay_buffer": None,
            "config": config,
        }

    total_timesteps = int(config["total_timesteps"])
    gamma = float(config["gamma"])
    learning_starts = int(config["learning_starts"])
    min_epsilon = float(config["min_epsilon"])
    log_every_steps = int(config["log_every_steps"])
    eval_every_steps = int(config["eval_every_steps"])
    epsilon_decay = float(config["epsilon_decay"])
    lr_decay = float(config["lr_decay"])
    q_init = float(config["q_init"])
    use_crm = bool(config["use_crm"])
    use_rs = bool(config["use_rs"])

    train_start = time.time()
    interrupted = False

    while global_step < total_timesteps:
        explore_delay = random.random() < epsilon_delay
        explore_action = random.random() < epsilon_action

        init_Q_delay_action(Q, s, delays, env_actions, q_init)
        best_delay, best_action = get_best_delay_action(
            Q, s, delays, env_actions, max_constant_array, discretization_param, num_clocks
        )
        if explore_delay or explore_action or global_step < learning_starts:
            a = random.choice(actions)
        else:
            a = (best_delay, best_action)

        action_index = env.env.action_to_index[a]
        sn, r, term, trunc, info = env.step(action_index)
        sn = tuple(sn)
        done = term or trunc
        t = discounted_time_from_action(a, discretization_param)

        if use_crm:
            experiences = [(s, sn, a, r, done, info)]
            for _s, _sn, _a_idx, _r, _done, _info in info["crm-experience"]:
                _s = tuple(_s)
                _sn = tuple(_sn)
                _a = tuple(env.env.index_to_action[_a_idx])
                experiences.append((_s, _sn, _a, _r, _done, _info))
        elif use_rs:
            experiences = [(s, sn, a, r, done, info)]
        else:
            experiences = [(s, sn, a, r, done, info)]

        for _s, _sn, _a, _r, _done, _info in experiences:
            init_Q_delay_action(Q, _s, delays, env_actions, q_init)
            init_Q_delay_action(Q, _sn, delays, env_actions, q_init)
            if _done:
                delta = _r - Q[_s][_a]
            else:
                _t = discounted_time_from_action(_a, discretization_param)
                delta = _r + (gamma ** _t) * get_qmax(Q, _sn, actions) - Q[_s][_a]
            Q[_s][_a] += lr * delta

        episode_reward += r
        episode_discounted_reward += (gamma ** episode_time) * r
        episode_time += t
        global_step += 1

        if done:
            obs, _ = env.reset(seed=seed, options={"random": False})
            s = tuple(obs)
            init_Q_delay_action(Q, s, delays, env_actions, q_init)
            running_reward = 0.05 * episode_reward + 0.95 * running_reward
            running_time = 0.05 * episode_time + 0.95 * running_time
            running_discounted_reward = 0.05 * episode_discounted_reward + 0.95 * running_discounted_reward
            current_episode_reward = episode_reward
            current_episode_reward_discounted = episode_discounted_reward
            current_episode_time = episode_time
            episode_reward = 0.0
            episode_time = 0.0
            episode_discounted_reward = 0.0
            num_episodes += 1
            if global_step > learning_starts:
                epsilon_delay = max(min_epsilon, epsilon_delay * epsilon_decay)
                epsilon_action = max(min_epsilon, epsilon_action * epsilon_decay)
                lr *= lr_decay
        else:
            s = sn

        if global_step % log_every_steps == 0:
            row = {
                "global_step": global_step,
                "episode": num_episodes,
                "running_reward": running_reward,
                "running_time": running_time,
                "running_discounted_reward": running_discounted_reward,
                "episode_reward": current_episode_reward,
                "episode_time": current_episode_time,
                "episode_discounted_reward": current_episode_reward_discounted,
                "epsilon_low": epsilon_delay,
                "epsilon_high": epsilon_action,
                "lr_low": lr,
                "lr_high": lr,
                "timestamp": time.time(),
            }
            metrics.log_train(row)
            writer.add_scalar("values/running_reward", running_reward, global_step)
            writer.add_scalar("values/running_time", running_time, global_step)
            writer.add_scalar("values/running_discounted_reward", running_discounted_reward, global_step)
            writer.add_scalar("values/episode_reward", current_episode_reward, global_step)
            writer.add_scalar("values/episode_discounted_reward", current_episode_reward_discounted, global_step)
            writer.add_scalar("values/episode_time", current_episode_time, global_step)
            init_Q_delay_action(Q, init_state, delays, env_actions, q_init)
            writer.add_scalar("values/init_q", get_qmax(Q, init_state, actions), global_step)

        if global_step % eval_every_steps == 0:
            eval_metrics = _evaluate_baseline_policy(
                eval_env,
                Q,
                q_init,
                gamma,
                eval_seed=seed + 100000 + eval_id,
            )
            metrics.log_eval(
                {
                    "global_step": global_step,
                    "eval_id": eval_id,
                    "episode_reward": eval_metrics["episode_reward"],
                    "episode_time": eval_metrics["episode_time"],
                    "episode_discounted_reward": eval_metrics["episode_discounted_reward"],
                    "timestamp": time.time(),
                }
            )
            writer.add_scalar("eval/episode_reward", eval_metrics["episode_reward"], global_step)
            writer.add_scalar(
                "eval/episode_discounted_reward",
                eval_metrics["episode_discounted_reward"],
                global_step,
            )
            writer.add_scalar("eval/episode_time", eval_metrics["episode_time"], global_step)
            eval_id += 1

        should_save, reason = ckpt.should_save(global_step, num_episodes)
        if should_save:
            ckpt.save(build_checkpoint_payload(reason), global_step, num_episodes, reason=reason)

        if terminator.stop_requested:
            interrupted = True
            ckpt.save(
                build_checkpoint_payload(f"signal_{terminator.signal_name}"),
                global_step,
                num_episodes,
                reason=f"signal_{terminator.signal_name}",
                force_milestone=True,
            )
            break

    elapsed = time.time() - train_start
    completed = (global_step >= total_timesteps) and not interrupted
    final_payload = build_checkpoint_payload("final" if completed else "interrupted")
    ckpt.save_final(final_payload, global_step, num_episodes)
    writer.close()
    terminator.restore()

    summary = {
        "algorithm": algorithm_name,
        "completed": completed,
        "interrupted": interrupted,
        "global_step": global_step,
        "episodes": num_episodes,
        "elapsed_seconds": elapsed,
        "seed": seed,
        "run_dir": str(run_dir),
    }
    save_json(run_dir / "summary.json", summary)
    return summary
