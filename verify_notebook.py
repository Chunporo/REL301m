import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt

class EpsilonGreedyAgent:
    def __init__(self, num_actions, epsilon=0.1):
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.action_values = np.zeros(num_actions)
        self.action_counts = np.zeros(num_actions)

    def select_action(self):
        if np.random.rand() < self.epsilon:
            # Randomly choose an action for exploration
            action = np.random.randint(self.num_actions)
        else:
            # Choose the greedy action for exploitation
            action = np.argmax(self.action_values)
        return action

    def update_value(self, action, reward):
        self.action_counts[action] += 1
        # Update action-value estimate using incremental update rule
        self.action_values[action] += (1 / self.action_counts[action]) * (reward - self.action_values[action])

# Create a simple multi-armed bandit environment
class MultiArmedBandit:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.true_action_values = np.random.normal(0, 1, num_arms)

    def get_reward(self, action):
        # Reward is sampled from a normal distribution with mean true action value and unit variance
        return np.random.normal(self.true_action_values[action], 1)

# Initialize the environment and agent
num_arms = 10
num_steps = 1000
agent = EpsilonGreedyAgent(num_arms)

# Interaction loop
bandit = MultiArmedBandit(num_arms)
total_rewards = 0
rewards_history = []
average_rewards = []

for step in range(num_steps):
    action = agent.select_action()
    reward = bandit.get_reward(action)
    agent.update_value(action, reward)

    total_rewards += reward
    rewards_history.append(reward)
    average_rewards.append(total_rewards / (step + 1))

print("Total rewards obtained:", total_rewards)
print("Estimated action values:", agent.action_values)

# Visualizations
plt.figure(figsize=(12, 5))

# Plot 1: Average Reward over Steps
plt.subplot(1, 2, 1)
plt.plot(average_rewards)
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.title('Average Reward over Time')
plt.grid(True)

# Plot 2: Estimated vs True Action Values
plt.subplot(1, 2, 2)
plt.plot(bandit.true_action_values, label='True Values', marker='o', linestyle='None')
plt.plot(agent.action_values, label='Estimated Values', marker='x', linestyle='None')
plt.xlabel('Action (Arm)')
plt.ylabel('Value')
plt.title('True vs Estimated Action Values')
plt.legend()
plt.grid(True)

plt.tight_layout()
print("Verification: Plot code executed successfully.")
