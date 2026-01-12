
# Imports
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

# --- Case Study 1: Toy Dynamics ---

# Original Code with Modifications
STATES = [-2, -1, 0, 1, 2]  # temperature error
A_OFF, A_ON = 0, 1

def step(e, a):
    noise = random.choice([-1, 0, 0, 0, 1])  # mostly 0
    if a == A_ON:
        e2 = e - 1 + noise
    else:
        e2 = e + 1 + noise
    e2 = max(-2, min(2, e2))
    r = -abs(e2)  # reward based on next state comfort
    return e2, r

def policy_threshold(e):
    return A_OFF if e == 0 else A_ON

def rollout_return(gamma, T=200, seed=0):
    random.seed(seed)
    e = random.choice(STATES)
    G = 0.0
    pow_g = 1.0
    for _ in range(T):
        a = policy_threshold(e)
        e, r = step(e, a)
        G += pow_g * r
        pow_g *= gamma
    return G

# Analysis Action
gammas = [0.0, 0.5, 0.9, 0.99]
results = []

for g in gammas:
    avg = sum(rollout_return(g, T=200, seed=s) for s in range(20)) / 20
    results.append({'Gamma': g, 'Avg Discounted Return': avg})

df_gamma = pd.DataFrame(results)
print("DataFrame for Case Study 1:")
print(df_gamma)

# Visualization
plt.figure(figsize=(8, 5))
sns.barplot(data=df_gamma, x='Gamma', y='Avg Discounted Return', palette='viridis')
plt.title("Average Discounted Return for Different Gamma Values")
plt.xlabel("Gamma (Discount Factor)")
plt.ylabel("Average Discounted Return")
plt.show()

# --- Case Study 2: Q-Learning ---

env = gym.make('CartPole-v1')

# Parameters
learning_rate = 0.1
discount_factor = 0.99
epsilon = 0.1
num_episodes = 50

state_space_bins = [6, 12, 12, 6]
q_table = np.random.uniform(low=-1, high=1, size=(state_space_bins[0], state_space_bins[1], state_space_bins[2], state_space_bins[3], env.action_space.n))

def discretize_state(state):
    state_bins = []
    for i in range(len(state)):
        bin_idx = np.digitize(state[i], np.linspace(-1, 1, state_space_bins[i]))
        state_bins.append(min(bin_idx, state_space_bins[i] - 1))
    return tuple(state_bins)

rewards_log = []

for episode in range(num_episodes):
    state, _ = env.reset()
    state = discretize_state(state)
    done = False
    total_reward = 0

    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, done, _, info = env.step(action)
        next_state = discretize_state(next_state)

        q_table[state + (action,)] = q_table[state + (action,)] + learning_rate * (reward + discount_factor * np.max(q_table[next_state]) - q_table[state + (action,)])

        state = next_state
        total_reward += reward

    rewards_log.append({'Episode': episode + 1, 'Total Reward': total_reward})

env.close()

df_rewards = pd.DataFrame(rewards_log)
print("\nFirst 5 rows of Q-Learning Rewards:")
print(df_rewards.head())

# Visualization
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_rewards, x='Episode', y='Total Reward')
plt.title("Q-Learning Training Performance")
plt.xlabel("Episode Number")
plt.ylabel("Total Reward")
plt.grid(True)
plt.show()
