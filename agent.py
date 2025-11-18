import gym
import numpy
import random
from os import system, name
from time import sleep

# Define function to clear console window.
def clear():
    if name == 'nt':
        _ = system('cls')
    else:
        _ = system('clear')

clear()

"""Setup"""

env = gym.make("Taxi-v3", render_mode="ansi")

# 500 состояний, 6 действий
q_table = numpy.zeros([env.observation_space.n, env.action_space.n])

training_episodes = 20000
display_episodes = 10

# Hyperparameters
alpha = 0.1   # Learning Rate
gamma = 0.6   # Discount Factor
epsilon = 0.1 # ε-greedy

all_epochs = []
all_penalties = []

"""Training the Agent"""

for i in range(training_episodes):
    # ✅ ВАЖНО: reset() → (state, info)
    state, info = env.reset()
    done = False
    penalties, reward = 0, 0

    while not done:
        # ε-greedy
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = numpy.argmax(q_table[state])

        # ✅ ВАЖНО: step() → 5 значений в новом API
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        old_value = q_table[state, action]
        next_max = numpy.max(q_table[next_state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state

    if i % 100 == 0:
        print(f"Episode: {i}")

print("Training finished.\n")

"""Display and evaluate agent's performance after Q-learning."""

total_epochs, total_penalties = 0, 0

for _ in range(display_episodes):
    state, info = env.reset()
    epochs, penalties, reward = 0, 0, 0
    done = False

    while not done:
        action = numpy.argmax(q_table[state])
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1

        clear()
        frame = env.render()  # в режиме "ansi" это строка с картой
        print(frame)
        print(f"Timestep: {epochs}")
        print(f"State: {state}")
        print(f"Action: {action}")
        print(f"Reward: {reward}")
        sleep(0.15)

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {display_episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / display_episodes}")
print(f"Average penalties per episode: {total_penalties / display_episodes}")
