import csv
import json
import os
import pickle
import random
import signal
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional in some envs
    torch = None


TRAIN_HEADERS = [
    "global_step",
    "episode",
    "running_reward",
    "running_time",
    "running_discounted_reward",
    "episode_reward",
    "episode_time",
    "episode_discounted_reward",
    "epsilon_low",
    "epsilon_high",
    "lr_low",
    "lr_high",
    "timestamp",
]

EVAL_HEADERS = [
    "global_step",
    "eval_id",
    "episode_reward",
    "episode_time",
    "episode_discounted_reward",
    "timestamp",
]


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(path, payload):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_json(path, default=None):
    if not Path(path).exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _atomic_pickle_dump(path, payload):
    target = Path(path)
    ensure_dir(target.parent)
    with tempfile.NamedTemporaryFile(delete=False, dir=target.parent, suffix=".tmp") as tmp:
        pickle.dump(payload, tmp, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_name = tmp.name
    target_abs = str(target.resolve())
    for _ in range(5):
        try:
            os.replace(tmp_name, target_abs)
            return
        except PermissionError:
            time.sleep(0.1)
    # Fallback for transient Windows lock behavior
    shutil.copyfile(tmp_name, target_abs)
    os.remove(tmp_name)


def load_pickle(path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    with open(p, "rb") as f:
        return pickle.load(f)


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                pass


def capture_rng_states():
    state = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
    }
    if torch is not None:
        state["torch_random_state"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda_random_state"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_states(state):
    if not state:
        return
    if "python_random_state" in state:
        random.setstate(state["python_random_state"])
    if "numpy_random_state" in state:
        np.random.set_state(state["numpy_random_state"])
    if torch is not None and "torch_random_state" in state:
        torch.set_rng_state(state["torch_random_state"])
        if torch.cuda.is_available() and "torch_cuda_random_state" in state:
            torch.cuda.set_rng_state_all(state["torch_cuda_random_state"])


def _to_python_number(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        if abs(f - round(f)) < 1e-10:
            return int(round(f))
        return f
    return v


def flatten_obs(obs):
    flat = np.asarray(obs).flatten().tolist()
    return tuple(_to_python_number(v) for v in flat)


def parse_product_observation(obs, feature_dim, num_clocks):
    flat = flatten_obs(obs)
    if len(flat) < feature_dim + 1 + num_clocks:
        raise ValueError(
            f"Unexpected observation size {len(flat)} (feature_dim={feature_dim}, num_clocks={num_clocks})"
        )
    s = tuple(flat[:feature_dim])
    u = int(flat[feature_dim])
    v = tuple(flat[feature_dim + 1 : feature_dim + 1 + num_clocks])
    return s, u, v, flat


def low_state_key(s, v):
    return (tuple(s), tuple(v))


def discounted_time_from_action(action, discretization_param):
    return action[0] * discretization_param + 1


def valid_augmented_actions(v, action_combinations, max_constant_array, discretization_param):
    v_arr = np.asarray(v, dtype=float)
    if discretization_param != 1:
        v_arr = v_arr * discretization_param
    slack = max_constant_array - v_arr
    max_possible_delay = float(np.max(slack))
    if discretization_param <= 0:
        max_delay_idx = 0
    else:
        max_delay_idx = int(np.floor(max_possible_delay / discretization_param + 1e-12))
    valid = [a for a in action_combinations if a[0] <= max_delay_idx]
    if valid:
        return valid
    zero_delay = [a for a in action_combinations if a[0] == 0]
    return zero_delay if zero_delay else list(action_combinations)


def build_successor_map(rm):
    succ = {}
    for u in rm.states:
        succ.setdefault(int(u), set())
    for u1, by_dnf in rm.outgoing_transitions.items():
        u1 = int(u1)
        succ.setdefault(u1, set())
        for transitions in by_dnf.values():
            for trans in transitions:
                u2 = int(trans[5])
                succ[u1].add(u2)
    return {u: sorted(list(vs)) for u, vs in succ.items()}


def valid_targets(successor_map, u):
    return [ut for ut in successor_map.get(int(u), []) if ut != int(u)]


class MetricsLogger:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        ensure_dir(self.run_dir)
        self.train_csv = self.run_dir / "train_metrics.csv"
        self.eval_csv = self.run_dir / "eval_metrics.csv"
        self.train_jsonl = self.run_dir / "train_metrics.jsonl"
        self.eval_jsonl = self.run_dir / "eval_metrics.jsonl"
        self._ensure_headers()

    def _ensure_headers(self):
        if not self.train_csv.exists():
            with open(self.train_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=TRAIN_HEADERS)
                writer.writeheader()
        if not self.eval_csv.exists():
            with open(self.eval_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=EVAL_HEADERS)
                writer.writeheader()
        for p in (self.train_jsonl, self.eval_jsonl):
            if not p.exists():
                p.touch()

    def _append_csv(self, path, headers, row):
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writerow({k: row.get(k, "") for k in headers})

    @staticmethod
    def _append_jsonl(path, row):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    def log_train(self, row):
        self._append_csv(self.train_csv, TRAIN_HEADERS, row)
        self._append_jsonl(self.train_jsonl, row)

    def log_eval(self, row):
        self._append_csv(self.eval_csv, EVAL_HEADERS, row)
        self._append_jsonl(self.eval_jsonl, row)


class CheckpointManager:
    def __init__(
        self,
        run_dir,
        save_every_steps=50000,
        save_every_episodes=25,
        save_every_seconds=900,
        save_milestones=False,
    ):
        self.run_dir = Path(run_dir)
        self.ckpt_dir = self.run_dir / "checkpoints"
        ensure_dir(self.ckpt_dir)
        self.save_every_steps = max(1, int(save_every_steps))
        self.save_every_episodes = max(1, int(save_every_episodes))
        self.save_every_seconds = max(1, int(save_every_seconds))
        self.save_milestones = bool(save_milestones)
        self.latest_path = self.ckpt_dir / "latest.pkl"
        self.final_path = self.ckpt_dir / "final.pkl"
        self._last_saved_step = -1
        self._last_saved_episode = -1
        self._last_saved_wall = time.time()

    def load_latest(self):
        return load_pickle(self.latest_path, default=None)

    def should_save(self, global_step, episode):
        now = time.time()
        if (global_step - self._last_saved_step) >= self.save_every_steps:
            return True, "step"
        if (episode - self._last_saved_episode) >= self.save_every_episodes:
            return True, "episode"
        if (now - self._last_saved_wall) >= self.save_every_seconds:
            return True, "wallclock"
        return False, ""

    def save(self, payload, global_step, episode, reason="manual", force_milestone=False):
        payload = dict(payload)
        payload["checkpoint_reason"] = reason
        payload["saved_at"] = time.time()
        _atomic_pickle_dump(self.latest_path, payload)
        make_milestone = self.save_milestones or force_milestone
        if make_milestone:
            milestone = self.ckpt_dir / f"step_{int(global_step):09d}.pkl"
            _atomic_pickle_dump(milestone, payload)
        self._last_saved_step = int(global_step)
        self._last_saved_episode = int(episode)
        self._last_saved_wall = time.time()

    def save_final(self, payload, global_step, episode):
        self.save(payload, global_step, episode, reason="final", force_milestone=True)
        _atomic_pickle_dump(self.final_path, payload)


class GracefulTerminator:
    def __init__(self):
        self.stop_requested = False
        self.signal_name = None
        self._old_int = signal.getsignal(signal.SIGINT)
        self._old_term = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signum, _frame):
        self.stop_requested = True
        self.signal_name = signal.Signals(signum).name

    def restore(self):
        signal.signal(signal.SIGINT, self._old_int)
        signal.signal(signal.SIGTERM, self._old_term)


def run_completed(summary_path):
    summary = load_json(summary_path, default={})
    return bool(summary.get("completed", False))
