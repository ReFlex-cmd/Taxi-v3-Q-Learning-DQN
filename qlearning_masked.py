import os
import json
import random
from typing import Dict, Any, List

import numpy as np
import gym


def train_q_learning(
    episodes: int = 5000,
    alpha: float = 0.1,
    gamma: float = 0.95,
    epsilon: float = 0.1,
    use_action_mask: bool = True,
    seed: int = 0,
    log_path: str | None = None,
) -> Dict[str, Any]:
    """
    Табличный Q-learning для Taxi-v3 с опциональным action-mask и логированием переходов.

    :param episodes: количество эпизодов обучения
    :param alpha: скорость обучения
    :param gamma: дисконт
    :param epsilon: параметр ε-greedy
    :param use_action_mask: использовать ли action_mask из info
    :param seed: базовый сид
    :param log_path: путь до .jsonl файла для логов переходов (или None, чтобы не логировать)
    :return: словарь с метриками и Q-таблицей
    """

    # ----- reproducibility -----
    np.random.seed(seed)
    random.seed(seed)

    # ВАЖНО: без .env — работаем с обёрнутым env (EnvChecker и т.п.)
    env = gym.make("Taxi-v3")
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    q_table = np.zeros((n_states, n_actions), dtype=np.float32)

    # ----- подготовка логирования переходов -----
    log_file = None
    if log_path is not None:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log_file = open(log_path, "w", encoding="utf-8")

    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    successes = 0

    for ep in range(episodes):
        # reset возвращает (obs, info)
        state, info = env.reset(seed=seed + ep)
        done = False
        truncated = False
        total_reward = 0.0
        steps = 0

        while not (done or truncated):
            # Вытаскиваем action_mask, если он есть
            if use_action_mask and "action_mask" in info:
                action_mask = np.array(info["action_mask"], dtype=np.int8)
            else:
                action_mask = np.ones(n_actions, dtype=np.int8)

            valid_actions = np.nonzero(action_mask == 1)[0]
            if len(valid_actions) == 0:
                # на всякий пожарный — если маска пустая, разрешаем всё
                valid_actions = np.arange(n_actions)

            # ε-greedy выбор действия с учётом маски
            if np.random.rand() < epsilon:
                action = int(np.random.choice(valid_actions))
            else:
                q_vals = q_table[state, valid_actions]
                action = int(valid_actions[int(np.argmax(q_vals))])

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += float(reward)
            steps += 1

            # next_action_mask для bootstrapping
            if use_action_mask and "action_mask" in info:
                next_mask = np.array(info["action_mask"], dtype=np.int8)
                valid_next_actions = np.nonzero(next_mask == 1)[0]
                if len(valid_next_actions) > 0:
                    next_max = float(np.max(q_table[next_state, valid_next_actions]))
                else:
                    next_max = 0.0
            else:
                next_max = float(np.max(q_table[next_state]))

            old_q = q_table[state, action]
            q_table[state, action] = (1 - alpha) * old_q + alpha * (
                reward + gamma * next_max
            )

            # логирование перехода
            if log_file is not None:
                rec = {
                    "episode": ep,
                    "t": steps,
                    "state": int(state),
                    "action": int(action),
                    "reward": float(reward),
                    "next_state": int(next_state),
                    "done": bool(done),
                    "action_mask": action_mask.tolist(),
                }
                log_file.write(json.dumps(rec, ensure_ascii=False) + "\n")

            state = next_state

        # терминация = успешная доставка пассажира
        if done and terminated:
            successes += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        if (ep + 1) % 100 == 0:
            last100 = episode_rewards[-100:]
            print(
                f"[seed={seed} mask={use_action_mask}] "
                f"Episode {ep + 1}/{episodes} | "
                f"meanR_100={np.mean(last100):.2f} | "
                f"success_rate={successes / (ep + 1):.2f}"
            )

    if log_file is not None:
        log_file.close()
    env.close()

    return {
        "q_table": q_table,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "success_rate": successes / episodes,
    }

def train_double_q_learning(
    episodes: int = 5000,
    alpha: float = 0.1,
    gamma: float = 0.95,
    epsilon: float = 0.1,
    use_action_mask: bool = True,
    seed: int = 0,
    log_path: str | None = None,
) -> Dict[str, Any]:
    """
    Double Q-learning для Taxi-v3 с опциональным action-mask и логированием переходов.

    Отличия от обычного Q-learning:
    - поддерживаются две таблицы Q1 и Q2;
    - на каждом шаге с вероятностью 0.5 обновляется либо Q1 (через argmax по Q1 и таргет из Q2),
      либо Q2 (через argmax по Q2 и таргет из Q1);
    - политика выбора действий строится по среднему Q = (Q1 + Q2) / 2.
    """
    np.random.seed(seed)
    random.seed(seed)

    env = gym.make("Taxi-v3")
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    q1 = np.zeros((n_states, n_actions), dtype=np.float32)
    q2 = np.zeros((n_states, n_actions), dtype=np.float32)

    log_file = None
    if log_path is not None:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log_file = open(log_path, "w", encoding="utf-8")

    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    successes = 0

    for ep in range(episodes):
        state, info = env.reset(seed=seed + ep)
        done = False
        truncated = False
        total_reward = 0.0
        steps = 0

        while not (done or truncated):
            # Маска действий
            if use_action_mask and "action_mask" in info:
                action_mask = np.array(info["action_mask"], dtype=np.int8)
            else:
                action_mask = np.ones(n_actions, dtype=np.int8)

            valid_actions = np.nonzero(action_mask == 1)[0]
            if len(valid_actions) == 0:
                valid_actions = np.arange(n_actions)

            # Политика строится по среднему Q
            q_avg = (q1[state] + q2[state]) / 2.0

            if np.random.rand() < epsilon:
                action = int(np.random.choice(valid_actions))
            else:
                q_vals = q_avg[valid_actions]
                action = int(valid_actions[int(np.argmax(q_vals))])

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += float(reward)
            steps += 1

            # Маска для next_state (для таргета)
            if use_action_mask and "action_mask" in info:
                next_mask = np.array(info["action_mask"], dtype=np.int8)
                valid_next = np.nonzero(next_mask == 1)[0]
                if len(valid_next) == 0:
                    valid_next = np.arange(n_actions)
            else:
                valid_next = np.arange(n_actions)

            # Обновляем либо Q1, либо Q2
            if np.random.rand() < 0.5:
                # обновляем Q1, используя argmax по Q1 и таргет из Q2
                a_star = int(valid_next[int(np.argmax(q1[next_state, valid_next]))])
                target = reward + (0.0 if done else gamma * float(q2[next_state, a_star]))
                old = q1[state, action]
                q1[state, action] = (1 - alpha) * old + alpha * target
            else:
                # обновляем Q2, используя argmax по Q2 и таргет из Q1
                a_star = int(valid_next[int(np.argmax(q2[next_state, valid_next]))])
                target = reward + (0.0 if done else gamma * float(q1[next_state, a_star]))
                old = q2[state, action]
                q2[state, action] = (1 - alpha) * old + alpha * target

            # Логирование перехода (как у обычного Q-learning)
            if log_file is not None:
                rec = {
                    "episode": ep,
                    "t": steps,
                    "state": int(state),
                    "action": int(action),
                    "reward": float(reward),
                    "next_state": int(next_state),
                    "done": bool(done),
                    "action_mask": action_mask.tolist(),
                }
                log_file.write(json.dumps(rec, ensure_ascii=False) + "\n")

            state = next_state

        if done and terminated:
            successes += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        if (ep + 1) % 100 == 0:
            last100 = episode_rewards[-100:]
            print(
                f"[DoubleQ seed={seed} mask={use_action_mask}] "
                f"Episode {ep + 1}/{episodes} | "
                f"meanR_100={np.mean(last100):.2f} | "
                f"success_rate={successes / (ep + 1):.2f}"
            )

    if log_file is not None:
        log_file.close()
    env.close()

    q_avg_final = (q1 + q2) / 2.0

    return {
        "q1_table": q1,
        "q2_table": q2,
        "q_table_avg": q_avg_final,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "success_rate": successes / episodes,
    }


def run_experiments():
    """
    Мини-оркестратор: гоняем Q-learning с mask ON/OFF и разными seed,
    пишем логи в data/trajectories/.
    """
    seeds = [0, 1, 2]
    episodes = 5000

    configs = [
        {"use_action_mask": True, "name": "mask_on"},
        {"use_action_mask": False, "name": "mask_off"},
    ]

    results: Dict[str, Any] = {}

    for cfg in configs:
        for seed in seeds:
            tag = f"{cfg['name']}_seed{seed}"
            log_path = os.path.join("data", "trajectories", f"qlearning_{tag}.jsonl")
            print(f"\n=== Run {tag} ===")

            metrics = train_q_learning(
                episodes=episodes,
                alpha=0.1,
                gamma=0.95,
                epsilon=0.1,
                use_action_mask=cfg["use_action_mask"],
                seed=seed,
                log_path=log_path,
            )

            results[tag] = {
                "success_rate": metrics["success_rate"],
                "mean_reward": float(np.mean(metrics["episode_rewards"])),
                "mean_length": float(np.mean(metrics["episode_lengths"])),
            }

    print("\n=== Summary over runs ===")
    for tag, m in results.items():
        print(
            f"{tag}: "
            f"success_rate={m['success_rate']:.3f}, "
            f"mean_reward={m['mean_reward']:.2f}, "
            f"mean_length={m['mean_length']:.2f}"
        )

def run_double_q_experiments():
    """
    Мини-оркестратор для Double Q-learning:
    mask ON/OFF, сиды 0/1/2, те же метрики и формат логов.
    """
    seeds = [0, 1, 2]
    episodes = 5000

    configs = [
        {"use_action_mask": True, "name": "mask_on"},
        {"use_action_mask": False, "name": "mask_off"},
    ]

    results: Dict[str, Any] = {}

    for cfg in configs:
        for seed in seeds:
            tag = f"doubleq_{cfg['name']}_seed{seed}"
            log_path = os.path.join("data", "trajectories", f"{tag}.jsonl")
            print(f"\n=== DoubleQ Run {tag} ===")

            metrics = train_double_q_learning(
                episodes=episodes,
                alpha=0.1,
                gamma=0.95,
                epsilon=0.1,
                use_action_mask=cfg["use_action_mask"],
                seed=seed,
                log_path=log_path,
            )

            results[tag] = {
                "success_rate": metrics["success_rate"],
                "mean_reward": float(np.mean(metrics["episode_rewards"])),
                "mean_length": float(np.mean(metrics["episode_lengths"])),
            }

    print("\n=== Double Q-learning summary over runs ===")
    for tag, m in results.items():
        print(
            f"{tag}: "
            f"success_rate={m['success_rate']:.3f}, "
            f"mean_reward={m['mean_reward']:.2f}, "
            f"mean_length={m['mean_length']:.2f}"
        )



if __name__ == "__main__":
    # Сначала стандартный Q-learning
    run_experiments()

    # Потом Double Q-learning (та же схема конфигов)
    run_double_q_experiments()
