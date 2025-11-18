import os
import json
from typing import Dict, Any, Tuple, List

import numpy as np
import gymnasium as gym
import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv


# ==========================
#  Обёртка: one-hot наблюдения
# ==========================

class OneHotObsEnv(gym.ObservationWrapper):
    """
    Преобразует дискретное состояние Taxi-v3 (0..499)
    в one-hot вектор длины n_states.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete)
        self.n_states = env.observation_space.n
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_states,),
            dtype=np.float32,
        )

    def observation(self, observation: int) -> np.ndarray:
        vec = np.zeros(self.n_states, dtype=np.float32)
        vec[observation] = 1.0
        return vec


def make_env(seed: int = 0):
    """Фабрика окружения Taxi-v3 с one-hot наблюдениями для SB3."""
    def _init():
        env = gym.make("Taxi-v3")
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env = OneHotObsEnv(env)
        return env
    return _init


# ==========================
#  DQN с тюнингом под Taxi-v3
# ==========================

def train_dqn_best(
    total_timesteps: int = 120_000,
    seed: int = 0,
) -> DQN:
    """
    DQN, настроенный под Taxi-v3:
    - one-hot наблюдения
    - MLP [64, 64]
    - более "такси-подобные" гиперпараметры
    """

    env = DummyVecEnv([make_env(seed)])

    policy_kwargs = dict(
        net_arch=[32, 32],
        activation_fn=th.nn.ReLU,
    )

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=5e-4,
        gamma=0.99,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=64,
        target_update_interval=500,
        train_freq=4,
        gradient_steps=1,
        exploration_fraction=0.4,
        exploration_final_eps=0.05,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        device="cpu",
    )

    model.learn(total_timesteps=total_timesteps)
    env.close()
    return model


def masked_action(model: DQN, obs: np.ndarray, action_mask: np.ndarray) -> int:
    """
    Выбор действия с учётом action-mask:
    Q недопустимых действий сильно занижаем перед argmax.
    """
    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    with th.no_grad():
        q_values = model.policy.q_net(obs_tensor)[0].cpu().numpy()

    q_masked = q_values.copy()
    invalid = (action_mask == 0)
    q_masked[invalid] = -1e9

    return int(np.argmax(q_masked))


def evaluate_dqn_best(
    model: DQN,
    use_action_mask: bool,
    n_episodes: int,
    seed: int,
    log_path: str | None = None,
) -> Dict[str, Any]:
    """
    Оценка dqn_taxi_best с/без action-mask.
    Наблюдения тоже one-hot, как при обучении.
    Переходы логируются в JSONL.
    """
    base_env = gym.make("Taxi-v3")
    env = OneHotObsEnv(base_env)

    # Логи
    log_file = None
    if log_path is not None:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log_file = open(log_path, "w", encoding="utf-8")

    rewards: List[float] = []
    lengths: List[int] = []
    successes = 0

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        truncated = False
        total_reward = 0.0
        steps = 0

        while not (done or truncated):
            # action_mask из info, если он есть
            if "action_mask" in info:
                action_mask = np.array(info["action_mask"], dtype=np.int8)
            else:
                action_mask = np.ones(env.action_space.n, dtype=np.int8)

            if use_action_mask:
                action = masked_action(model, obs, action_mask)
            else:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += float(reward)
            steps += 1

            # из one-hot получаем индекс состояния (для логов)
            state_idx = int(np.argmax(obs))
            next_state_idx = int(np.argmax(next_obs))

            if log_file is not None:
                rec = {
                    "episode": ep,
                    "t": steps,
                    "state": state_idx,
                    "action": int(action),
                    "reward": float(reward),
                    "next_state": next_state_idx,
                    "done": bool(done),
                    "action_mask": action_mask.tolist(),
                }
                log_file.write(json.dumps(rec, ensure_ascii=False) + "\n")

            obs = next_obs

        if done and terminated:
            successes += 1

        rewards.append(total_reward)
        lengths.append(steps)

    if log_file is not None:
        log_file.close()
    env.close()

    return {
        "success_rate": successes / n_episodes,
        "mean_reward": float(np.mean(rewards)),
        "mean_length": float(np.mean(lengths)),
    }


def run_dqn_taxi_best():
    """
    Один “best” конфиг:
    - архитектура [64, 64]
    - one-hot наблюдения
    - mask=ON (основной режим) + для сравнения mask=OFF
    - несколько сидов
    """

    seeds = [0, 1, 2]
    total_timesteps = 120_000
    n_eval_episodes = 200

    summary: Dict[str, Any] = {}

    for seed in seeds:
        print(f"\n=== Training dqn_taxi_best seed={seed} ===")
        model = train_dqn_best(total_timesteps=total_timesteps, seed=seed)

        for use_mask in [True, False]:
            tag = f"dqn_taxi_best_seed{seed}_mask_{use_mask}"
            log_path = os.path.join("data", "trajectories", f"{tag}.jsonl")
            print(f"\n--- Evaluating {tag} ---")
            metrics = evaluate_dqn_best(
                model=model,
                use_action_mask=use_mask,
                n_episodes=n_eval_episodes,
                seed=seed,
                log_path=log_path,
            )
            summary[tag] = metrics

    print("\n=== dqn_taxi_best summary ===")
    for tag, m in summary.items():
        print(
            f"{tag}: "
            f"success_rate={m['success_rate']:.3f}, "
            f"mean_reward={m['mean_reward']:.2f}, "
            f"mean_length={m['mean_length']:.2f}"
        )


if __name__ == "__main__":
    run_dqn_taxi_best()
