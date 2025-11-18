import os
import json
from typing import Dict, Any, Tuple, List

import numpy as np
import gymnasium as gym
import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv


def make_env(seed: int = 0):
    """Фабрика окружения для SB3 (используем Gymnasium)."""
    def _init():
        env = gym.make("Taxi-v3")
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env
    return _init


def train_dqn(
    net_arch: Tuple[int, int] = (32, 32),   # <-- фиксируем одну архитектуру
    total_timesteps: int = 60_000,        # <-- даём DQN больше шансов, чем 40k
    seed: int = 0,
) -> DQN:
    """
    Обучение DQN на Taxi-v3.

    :param net_arch: архитектура MLP, по факту (64, 64)
    :param total_timesteps: количество шагов обучения
    :param seed: сид
    :return: обученная модель DQN
    """
    env = DummyVecEnv([make_env(seed)])

    policy_kwargs = dict(
        net_arch=list(net_arch),
        activation_fn=th.nn.ReLU,
    )

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=5e-4,          # было 1e-3
        gamma=0.95,                  # ближе к табличному агенту (0.95)
        buffer_size=50_000,          # можно меньше, чем 100k
        batch_size=64,
        learning_starts=1_000,       # чуть опыта до первых апдейтов
        target_update_interval=500,  # почаще обновляем target
        train_freq=4,                # апдейты реже, но пачками
        gradient_steps=1,
        exploration_fraction=0.4,    # дольше исследуем
        exploration_final_eps=0.05,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
    )

    model.learn(total_timesteps=total_timesteps)
    env.close()
    return model


def masked_action(model: DQN, obs: np.ndarray, action_mask: np.ndarray) -> int:
    """
    Выбор действия с учётом маски: зануляем (точнее, занижаем) Q недопустимых действий.
    """
    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    with th.no_grad():
        q_values = model.policy.q_net(obs_tensor)[0].cpu().numpy()

    q_masked = q_values.copy()
    invalid = (action_mask == 0)
    q_masked[invalid] = -1e9  # сильно штрафуем запрещённые действия

    return int(np.argmax(q_masked))


def evaluate_dqn(
    model: DQN,
    use_action_mask: bool,
    n_episodes: int,
    seed: int,
    log_path: str | None = None,
) -> Dict[str, Any]:
    """
    Оценка DQN в Taxi-v3 с/без action-mask + логирование переходов.

    :param model: обученный DQN
    :param use_action_mask: использовать ли mask при выборе действий
    :param n_episodes: количество эпизодов оценки
    :param seed: базовый сид
    :param log_path: путь к .jsonl файлу для логов переходов
    :return: словарь с метриками
    """
    env = gym.make("Taxi-v3")

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
            # Достаём маску (если есть)
            if "action_mask" in info:
                action_mask = np.array(info["action_mask"], dtype=np.int8)
            else:
                action_mask = np.ones(env.action_space.n, dtype=np.int8)


            if use_action_mask:
                action = masked_action(model, obs, action_mask)
            else:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)  # <-- ВАЖНО: превращаем из np.ndarray в обычный int

            next_obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            total_reward += float(reward)
            steps += 1

            if log_file is not None:
                rec = {
                    "episode": ep,
                    "t": steps,
                    "state": int(obs),
                    "action": int(action),
                    "reward": float(reward),
                    "next_state": int(next_obs),
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


def run_dqn_experiments():
    """
    Запускает эксперименты DQN на одной архитектуре (64–64)
    с mask ON/OFF и несколькими сидами.
    """
    seeds = [0, 1, 2]
    net_archs = [
        ((32, 32), "32-32_2"),   # <-- только одна архитектура
    ]
    n_train_steps = 60_000    # <-- согласовано с train_dqn()
    n_eval_episodes = 200

    summary: Dict[str, Any] = {}

    for net_arch, arch_name in net_archs:
        for seed in seeds:
            print(f"\n=== Training DQN arch={arch_name} seed={seed} ===")
            model = train_dqn(net_arch=net_arch, total_timesteps=n_train_steps, seed=seed)

            for use_mask in [True, False]:
                tag = f"dqn_arch_{arch_name}_seed{seed}_mask_{use_mask}"
                log_path = os.path.join("data", "trajectories", f"{tag}.jsonl")
                print(f"\n--- Evaluating {tag} ---")
                metrics = evaluate_dqn(
                    model=model,
                    use_action_mask=use_mask,
                    n_episodes=n_eval_episodes,
                    seed=seed,
                    log_path=log_path,
                )
                summary[tag] = metrics

    print("\n=== DQN summary ===")
    for tag, m in summary.items():
        print(
            f"{tag}: "
            f"success_rate={m['success_rate']:.3f}, "
            f"mean_reward={m['mean_reward']:.2f}, "
            f"mean_length={m['mean_length']:.2f}"
        )


if __name__ == "__main__":
    run_dqn_experiments()
