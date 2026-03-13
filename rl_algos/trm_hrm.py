import random
import time
from pathlib import Path

import numpy as np

from avg_tb import SummaryWriter
from rl_algos.trm_experiment_utils import (
    CheckpointManager,
    GracefulTerminator,
    MetricsLogger,
    build_successor_map,
    capture_rng_states,
    discounted_time_from_action,
    low_state_key,
    parse_product_observation,
    restore_rng_states,
    save_json,
    set_global_seed,
    valid_augmented_actions,
    valid_targets,
)


def _ensure_state_action_table(table, state, actions, q_init):
    if state not in table:
        table[state] = {a: float(q_init) for a in actions}
    else:
        for a in actions:
            table[state].setdefault(a, float(q_init))


def _argmax_action(table, state, actions, q_init):
    _ensure_state_action_table(table, state, actions, q_init)
    qmax = max(table[state][a] for a in actions)
    best = [a for a in actions if table[state][a] == qmax]
    return random.choice(best), qmax


def _ensure_controller_state(Q_controller, high_state, target_states, q_init):
    if high_state not in Q_controller:
        Q_controller[high_state] = {u_tgt: float(q_init) for u_tgt in target_states}
    else:
        for u_tgt in target_states:
            Q_controller[high_state].setdefault(u_tgt, float(q_init))


def _controller_qmax(Q_controller, high_state, target_states, q_init):
    _ensure_controller_state(Q_controller, high_state, target_states, q_init)
    return max(Q_controller[high_state][u_tgt] for u_tgt in target_states)


def _greedy_controller_action(Q_controller, high_state, target_states, q_init):
    _ensure_controller_state(Q_controller, high_state, target_states, q_init)
    qmax = _controller_qmax(Q_controller, high_state, target_states, q_init)
    best = [u_tgt for u_tgt in target_states if Q_controller[high_state][u_tgt] == qmax]
    return random.choice(best)


def _get_option_table(Q_options, option_key):
    if option_key not in Q_options:
        Q_options[option_key] = {}
    return Q_options[option_key]


def _local_option_reward(r_product, u_source, u_target, u_next, r_plus, r_minus):
    if u_next == u_target:
        return r_product + r_plus
    if u_next not in {u_source, u_target}:
        return r_product + r_minus
    return r_product


def _evaluate_hrm_policy(
    env,
    Q_controller,
    Q_options,
    successor_map,
    q_init,
    gamma,
    feature_dim,
    num_clocks,
    max_constant_array,
    discretization_param,
    eval_seed,
    max_eval_steps=2000,
):
    all_aug_actions = env.env.action_combinations
    obs, _ = env.reset(seed=eval_seed, options={"random": False})
    s, u, v, high_state = parse_product_observation(obs, feature_dim, num_clocks)
    done = False
    episode_reward = 0.0
    episode_time = 0.0
    episode_discounted_reward = 0.0
    option_target = None
    option_source = None
    steps = 0

    while not done and steps < max_eval_steps:
        targets = valid_targets(successor_map, u)
        if option_target is None:
            if not targets:
                valid_actions = valid_augmented_actions(
                    v, all_aug_actions, max_constant_array, discretization_param
                )
                chosen_action = valid_actions[0]
                option_source = u
            else:
                option_target = _greedy_controller_action(
                    Q_controller, high_state, targets, q_init
                )
                option_source = u
                option_key = (option_source, option_target)
                low_s = low_state_key(s, v)
                option_table = _get_option_table(Q_options, option_key)
                valid_actions = valid_augmented_actions(
                    v, all_aug_actions, max_constant_array, discretization_param
                )
                chosen_action, _ = _argmax_action(option_table, low_s, valid_actions, q_init)
        else:
            option_key = (option_source, option_target)
            low_s = low_state_key(s, v)
            option_table = _get_option_table(Q_options, option_key)
            valid_actions = valid_augmented_actions(
                v, all_aug_actions, max_constant_array, discretization_param
            )
            chosen_action, _ = _argmax_action(option_table, low_s, valid_actions, q_init)

        action_index = env.env.action_to_index[chosen_action]
        sn, r, term, trunc, _ = env.step(action_index)
        done = term or trunc
        t = discounted_time_from_action(chosen_action, discretization_param)

        episode_reward += r
        episode_discounted_reward += (gamma ** episode_time) * r
        episode_time += t

        s_next, u_next, v_next, high_state_next = parse_product_observation(
            sn, feature_dim, num_clocks
        )
        if option_target is not None and (done or u_next != option_source):
            option_target = None
            option_source = None

        s, u, v, high_state = s_next, u_next, v_next, high_state_next
        steps += 1

    return {
        "episode_reward": float(episode_reward),
        "episode_time": float(episode_time),
        "episode_discounted_reward": float(episode_discounted_reward),
    }


def train_trm_hrm(
    env,
    eval_env,
    run_dir,
    config,
    resume=False,
):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "config.json", config)

    algorithm_name = str(config.get("algorithm", "trm_hrm"))

    metrics = MetricsLogger(run_dir)
    ckpt = CheckpointManager(
        run_dir=run_dir,
        save_every_steps=config["checkpoint_every_steps"],
        save_every_episodes=config["checkpoint_every_episodes"],
        save_every_seconds=int(config["checkpoint_every_minutes"] * 60),
        save_milestones=config["save_milestones"],
    )
    terminator = GracefulTerminator()
    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    seed = int(config["seed"])
    set_global_seed(seed)

    feature_dim = int(np.asarray(env.env.features_space.shape).prod())
    num_clocks = int(env.env.num_clocks)
    discretization_param = float(env.env.discretization_param)
    max_constant_array = np.asarray(list(env.env.max_constant_dict.values()), dtype=float)
    all_aug_actions = env.env.action_combinations
    successor_map = build_successor_map(env.env.reward_machines[0])

    Q_controller = {}
    Q_options = {}
    global_step = 0
    num_episodes = 0
    eval_id = 0

    epsilon_low = float(config["epsilon"])
    epsilon_high = float(config["epsilon"])
    lr_low = float(config["lr"])
    lr_high = float(config["hrm_lr"])

    running_reward = 0.0
    running_time = 0.0
    running_discounted_reward = 0.0
    current_episode_reward = 0.0
    current_episode_reward_discounted = 0.0
    current_episode_time = 0.0

    episode_reward = 0.0
    episode_time = 0.0
    episode_discounted_reward = 0.0

    obs, _ = env.reset(seed=seed, options={"random": False})
    s, u, v, high_state = parse_product_observation(obs, feature_dim, num_clocks)
    init_state = high_state

    # Active option context
    option_source = None
    option_target = None
    option_start_state = None
    option_return = 0.0
    option_elapsed = 0.0

    if resume:
        state = ckpt.load_latest()
        if state is not None:
            Q_controller = state["Q_controller"]
            Q_options = state["Q_options"]
            global_step = int(state["global_step"])
            num_episodes = int(state["num_episodes"])
            eval_id = int(state["eval_id"])
            epsilon_low = float(state["epsilon_low"])
            epsilon_high = float(state["epsilon_high"])
            lr_low = float(state["lr_low"])
            lr_high = float(state["lr_high"])
            running_reward = float(state["running_reward"])
            running_time = float(state["running_time"])
            running_discounted_reward = float(state["running_discounted_reward"])
            current_episode_reward = float(state["current_episode_reward"])
            current_episode_reward_discounted = float(state["current_episode_reward_discounted"])
            current_episode_time = float(state["current_episode_time"])
            episode_reward = float(state["episode_reward"])
            episode_time = float(state["episode_time"])
            episode_discounted_reward = float(state["episode_discounted_reward"])
            option_source = state.get("option_source")
            option_target = state.get("option_target")
            option_start_state = tuple(state["option_start_state"]) if state.get("option_start_state") else None
            option_return = float(state.get("option_return", 0.0))
            option_elapsed = float(state.get("option_elapsed", 0.0))
            restore_rng_states(state.get("rng_state"))
            obs, _ = env.reset(seed=seed, options={"random": False})
            s, u, v, high_state = parse_product_observation(obs, feature_dim, num_clocks)
            init_state = tuple(state.get("init_state", high_state))
            # Continue from a clean episode after resume.
            option_source = None
            option_target = None
            option_start_state = None
            option_return = 0.0
            option_elapsed = 0.0

    def build_checkpoint_payload(reason):
        return {
            "algorithm": algorithm_name,
            "reason": reason,
            "Q_controller": Q_controller,
            "Q_options": Q_options,
            "global_step": global_step,
            "num_episodes": num_episodes,
            "eval_id": eval_id,
            "epsilon_low": epsilon_low,
            "epsilon_high": epsilon_high,
            "lr_low": lr_low,
            "lr_high": lr_high,
            "running_reward": running_reward,
            "running_time": running_time,
            "running_discounted_reward": running_discounted_reward,
            "current_episode_reward": current_episode_reward,
            "current_episode_reward_discounted": current_episode_reward_discounted,
            "current_episode_time": current_episode_time,
            "episode_reward": episode_reward,
            "episode_time": episode_time,
            "episode_discounted_reward": episode_discounted_reward,
            "option_source": option_source,
            "option_target": option_target,
            "option_start_state": option_start_state,
            "option_return": option_return,
            "option_elapsed": option_elapsed,
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
    hrm_lr_decay = float(config.get("hrm_lr_decay", lr_decay))
    q_init = float(config["q_init"])
    r_plus = float(config["r_plus"])
    r_minus = float(config["r_minus"])
    parallel_option_updates = bool(config["parallel_option_updates"])
    use_crm = bool(config["use_crm"])

    train_start = time.time()
    interrupted = False

    while global_step < total_timesteps:
        targets = valid_targets(successor_map, u)

        if option_target is None:
            if targets:
                option_source = u
                option_start_state = high_state
                option_return = 0.0
                option_elapsed = 0.0
                _ensure_controller_state(Q_controller, option_start_state, targets, q_init)
                if random.random() < epsilon_high or global_step < learning_starts:
                    option_target = random.choice(targets)
                else:
                    option_target = _greedy_controller_action(
                        Q_controller,
                        option_start_state,
                        targets,
                        q_init,
                    )
            else:
                option_source = u
                option_target = None
                option_start_state = None

        valid_actions = valid_augmented_actions(
            v, all_aug_actions, max_constant_array, discretization_param
        )
        explore_low = (random.random() < epsilon_low) or (global_step < learning_starts)

        if option_target is None:
            chosen_action = random.choice(valid_actions)
        else:
            option_key = (option_source, option_target)
            option_table = _get_option_table(Q_options, option_key)
            low_s = low_state_key(s, v)
            if explore_low:
                chosen_action = random.choice(valid_actions)
            else:
                chosen_action, _ = _argmax_action(option_table, low_s, valid_actions, q_init)

        action_index = env.env.action_to_index[chosen_action]
        sn, r, term, trunc, info = env.step(action_index)
        done = term or trunc
        t = discounted_time_from_action(chosen_action, discretization_param)

        s_next, u_next, v_next, high_state_next = parse_product_observation(
            sn, feature_dim, num_clocks
        )

        episode_reward += r
        episode_discounted_reward += (gamma ** episode_time) * r
        episode_time += t
        global_step += 1

        # Low-level updates from the real transition and optional CI transitions.
        transition_batch = [(s, u, v, chosen_action, r, done, s_next, u_next, v_next)]
        if use_crm:
            for cf_obs, cf_next_obs, cf_action_idx, cf_reward, cf_done, _ in info.get("crm-experience", []):
                cf_s, cf_u, cf_v, _ = parse_product_observation(
                    cf_obs, feature_dim, num_clocks
                )
                cf_s_next, cf_u_next, cf_v_next, _ = parse_product_observation(
                    cf_next_obs, feature_dim, num_clocks
                )
                cf_action = tuple(env.env.index_to_action[int(cf_action_idx)])
                transition_batch.append(
                    (
                        cf_s,
                        cf_u,
                        cf_v,
                        cf_action,
                        float(cf_reward),
                        bool(cf_done),
                        cf_s_next,
                        cf_u_next,
                        cf_v_next,
                    )
                )

        for (
            src_s,
            src_u,
            src_v,
            src_action,
            src_r,
            src_done,
            dst_s,
            dst_u,
            dst_v,
        ) in transition_batch:
            source_targets = valid_targets(successor_map, src_u)
            if parallel_option_updates:
                update_targets = source_targets
            elif option_target is not None and option_source == src_u:
                update_targets = [option_target]
            else:
                update_targets = []

            for target_u in update_targets:
                option_key = (src_u, target_u)
                option_table = _get_option_table(Q_options, option_key)
                low_s = low_state_key(src_s, src_v)
                _ensure_state_action_table(option_table, low_s, all_aug_actions, q_init)
                r_loc = _local_option_reward(src_r, src_u, target_u, dst_u, r_plus, r_minus)
                option_terminated = src_done or (dst_u != src_u)
                if option_terminated:
                    y_low = r_loc
                else:
                    next_low_s = low_state_key(dst_s, dst_v)
                    next_valid_actions = valid_augmented_actions(
                        dst_v, all_aug_actions, max_constant_array, discretization_param
                    )
                    _ensure_state_action_table(
                        option_table, next_low_s, next_valid_actions, q_init
                    )
                    qmax_next = max(option_table[next_low_s][a] for a in next_valid_actions)
                    t_transition = discounted_time_from_action(
                        src_action, discretization_param
                    )
                    y_low = r_loc + (gamma ** t_transition) * qmax_next
                delta_low = y_low - option_table[low_s][src_action]
                option_table[low_s][src_action] += lr_low * delta_low

        # High-level SMDP update only when active option terminates.
        if option_target is not None:
            option_return += (gamma ** option_elapsed) * r
            option_elapsed += t

            option_terminated = done or (u_next != option_source)
            if option_terminated:
                _ensure_controller_state(
                    Q_controller, option_start_state, [option_target], q_init
                )
                end_targets = [] if done else valid_targets(successor_map, u_next)
                if done or not end_targets:
                    y_high = option_return
                else:
                    qmax_next = _controller_qmax(
                        Q_controller, high_state_next, end_targets, q_init
                    )
                    y_high = option_return + (gamma ** option_elapsed) * qmax_next
                delta_high = y_high - Q_controller[option_start_state][option_target]
                Q_controller[option_start_state][option_target] += lr_high * delta_high

                option_source = None
                option_target = None
                option_start_state = None
                option_return = 0.0
                option_elapsed = 0.0

        if done:
            obs, _ = env.reset(seed=seed, options={"random": False})
            s, u, v, high_state = parse_product_observation(obs, feature_dim, num_clocks)
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
            option_source = None
            option_target = None
            option_start_state = None
            option_return = 0.0
            option_elapsed = 0.0
            if global_step > learning_starts:
                epsilon_low = max(min_epsilon, epsilon_low * epsilon_decay)
                epsilon_high = max(min_epsilon, epsilon_high * epsilon_decay)
                lr_low *= lr_decay
                lr_high *= hrm_lr_decay
        else:
            s, u, v, high_state = s_next, u_next, v_next, high_state_next

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
                "epsilon_low": epsilon_low,
                "epsilon_high": epsilon_high,
                "lr_low": lr_low,
                "lr_high": lr_high,
                "timestamp": time.time(),
            }
            metrics.log_train(row)
            writer.add_scalar("values/running_reward", running_reward, global_step)
            writer.add_scalar("values/running_time", running_time, global_step)
            writer.add_scalar("values/running_discounted_reward", running_discounted_reward, global_step)
            writer.add_scalar("values/episode_reward", current_episode_reward, global_step)
            writer.add_scalar("values/episode_discounted_reward", current_episode_reward_discounted, global_step)
            writer.add_scalar("values/episode_time", current_episode_time, global_step)

            init_u = int(init_state[feature_dim]) if len(init_state) > feature_dim else None
            init_targets = valid_targets(successor_map, init_u) if init_u is not None else []
            if init_targets:
                _ensure_controller_state(Q_controller, init_state, init_targets, q_init)
                writer.add_scalar(
                    "values/init_q_controller",
                    _controller_qmax(Q_controller, init_state, init_targets, q_init),
                    global_step,
                )

        if global_step % eval_every_steps == 0:
            eval_metrics = _evaluate_hrm_policy(
                eval_env,
                Q_controller,
                Q_options,
                successor_map,
                q_init,
                gamma,
                feature_dim,
                num_clocks,
                max_constant_array,
                discretization_param,
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
        "parallel_option_updates": parallel_option_updates,
    }
    save_json(run_dir / "summary.json", summary)
    return summary
