# Research Summary: Assignment 2 - Function Approximation and Control

## 1. Overview
This notebook is a practical exercise on **Reinforcement Learning (RL)**, focusing on using **Function Approximation** to solve a Control problem. Specifically, it uses the **Sarsa** algorithm combined with **Tile Coding** in the **Mountain Car** environment.

**Main Objectives:**
*   Use function approximation (tile coding) in a control problem.
*   Implement the Sarsa algorithm.
*   Compare the effectiveness of different tile coding configurations.

## 2. Key Components

### A. Environment: Mountain Car
*   **Goal:** Control an under-powered car to reach the top of a hill. The car needs to build momentum (rock back and forth) to reach the goal.
*   **State Space:** Continuous space consisting of `position` (from -1.2 to 0.5) and `velocity` (from -0.07 to 0.07).
*   **Challenge:** Since the state space is continuous, a standard Q-table cannot be used, requiring Function Approximation (specifically Tile Coding).

### B. Tile Coding (`MountainCarTileCoder` Class)
The code is pre-implemented ("Lecturer" version) to handle converting continuous states into feature vectors:
*   Uses the `tiles3` library.
*   **Input Normalization:** `position` and `velocity` are scaled based on the number of tiles to match the encoder's input requirements.
*   **`get_tiles` Function:** Returns the indices of the active tiles based on the current state.

### C. Sarsa Agent (`SarsaAgent` Class)
This is the core component executing the RL algorithm. Key functions implemented include:
1.  **`select_action`**: Selects an action using an **epsilon-greedy** strategy.
    *   The Q-value of an action is calculated as the sum of weights (`self.w`) corresponding to the active tiles: $Q(s, a) = \sum w_i$.
2.  **`agent_start`**: Initializes the agent and selects the first action.
3.  **`agent_step`**: Updates weights using the Sarsa Semi-gradient formula:
    $$ \mathbf{w} \leftarrow \mathbf{w} + \alpha [R + \gamma \hat{q}(S', A', \mathbf{w}) - \hat{q}(S, A, \mathbf{w})] \nabla \hat{q}(S, A, \mathbf{w}) $$
    In Tile Coding, the gradient $\nabla \hat{q}$ is simply a vector of 1s at active tile positions and 0s elsewhere.
4.  **`agent_end`**: Updates the final step of the episode (when the terminal state has no future value).

## 3. Experiments and Results
The notebook compares the Agent's performance with 3 different Tile Coding configurations:
1.  **8 tilings, 8x8 grid**: Baseline configuration.
2.  **2 tilings, 16x16 grid**: Higher grid resolution but fewer tilings.
3.  **32 tilings, 4x4 grid**: Lower grid resolution but more tilings.

**Conclusions from the notebook:**
*   The **32 tilings (4x4)** configuration yielded the best results.
*   The **8 tilings (8x8)** configuration was second best.
*   The **2 tilings (16x16)** configuration performed the worst.
$\rightarrow$ This demonstrates that increasing the number of tilings generally helps the agent learn better and smoother (better generalization) than just increasing the resolution of individual grids.

## 4. Code Logic Sample
Below is the core logic of the update function in `agent_step` found in the notebook:

```python
# Get active tiles for the current state
active_tiles = self.tc.get_tiles(position, velocity)
current_action, action_value = self.select_action(active_tiles)

# Calculate Q-value for the previous state-action pair
last_action_value = np.sum(self.w[self.last_action][self.previous_tiles])

# Update weights (Sarsa update)
# w += alpha * (Reward + gamma * Q_next - Q_old) * gradient
self.w[self.last_action][self.previous_tiles] += self.alpha * (reward + self.gamma * action_value - last_action_value)
```
