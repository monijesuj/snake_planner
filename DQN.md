# Snake Game Planning with Deep Q-Networks (DQN)

This repository presents a Deep Reinforcement Learning–based planner for the classic Snake game, implemented as part of a graduate-level planning algorithms project.
The learned policy is compared against classical planners (A*, Dijkstra, BFS, RRT) in a dynamic environment where the snake’s body itself acts as a moving obstacle.

---

## Overview

The Snake game is formulated as a **Markov Decision Process (MDP)** and solved using a **Deep Q-Network (DQN)** with:

* A compact **11-dimensional state representation**
* A **relative action space** (straight, left, right)
* **Experience replay**
* **Target network stabilization**
* **ε-greedy exploration strategy**

The trained agent learns a policy that allows it to avoid collisions, navigate efficiently, and collect food consistently.

---

## Problem Formulation as an MDP

The environment is modeled as a finite MDP:

$$
\mathcal{M} = (S, A, P, R, \gamma)
$$

where:

* $( S )$ is the state space
* $( A )$ is the action space
* $( P(s' \mid s, a) )$ is the transition function
* $( R(s, a) )$ is the reward function
* $( \gamma \in [0,1] )$ is the discount factor

---

### State Space (S)

Each state is represented by an **11-dimensional vector**:

| Index | Feature                        | Description                             |
| ----: | ------------------------------ | --------------------------------------- |
|   1–3 | Danger Straight / Right / Left | Collision risk in relative directions   |
|   4–7 | Direction Flags                | Current movement direction (L, R, U, D) |
|  8–11 | Food Direction                 | Relative position of food               |

This representation is **grid-size invariant**, enabling generalization across different board sizes.

---

### Action Space (A)

The agent selects one of **three relative actions**:

| Action | Meaning     |
| ------ | ----------- |
| 0      | Go straight |
| 1      | Turn right  |
| 2      | Turn left   |

These relative actions are internally mapped to absolute grid directions based on the snake’s current heading.

---

### Reward Function (R)

The reward signal is defined as:

| Event                 | Reward |
| --------------------- | ------ |
| Eating food           | +10    |
| Collision (wall/body) | −10    |
| Step penalty          | −0.1   |

Formally:

$$
R(s, a) =
\begin{cases}
+10, & \text{if food is eaten} \\
-10, & \text{if collision occurs} \\
-0.1, & \text{otherwise}
\end{cases}
$$

This encourages survival and efficient food collection while discouraging unsafe actions.

---

### Transition Dynamics (P)

State transitions are **deterministic** and governed by:

* Snake movement rules
* Collision detection
* Food respawning logic

---

## Deep Q-Learning

### Q-Function

The action-value function is defined as:

$$
Q^\pi(s, a) = \mathbb{E}*\pi \left[ \sum*{t=0}^{\infty} \gamma^t r_{t} ,\middle|, s_0 = s, a_0 = a \right]
$$

The optimal Q-function satisfies the **Bellman Optimality Equation**:

$$
Q^*(s,a) = R(s,a) + \gamma \max_{a'} Q^*(s', a')
$$

---

### Network Architecture

The Q-function is approximated using a fully connected neural network:

```
Q-Network
Input (11)
  ↓
Linear (11 → 256) + ReLU
  ↓
Linear (256 → 256) + ReLU
  ↓
Linear (256 → 3)
```

* Outputs Q-values for the **3 relative actions**
* Trained using **Mean Squared Error (MSE)** loss

Loss function:

$$
\mathcal{L}(\theta) = \mathbb{E}*{(s,a,r,s')} \left[
\left(
r + \gamma \max*{a'} Q_{\theta^-}(s', a') - Q_{\theta}(s,a)
\right)^2
\right]
$$

---

### Target Network

A **separate target network** $( Q_{\theta^-} )$ is maintained to stabilize training.

* Parameters are periodically copied from the online Q-network:
  $
  \theta^- \leftarrow \theta
  $
* Prevents harmful feedback loops and oscillations during training

Target computation:

$$
y = r + \gamma \max_{a'} Q_{\theta^-}(s', a')
$$

---

### Experience Replay

To break temporal correlations and improve data efficiency:

* Transitions $( (s, a, r, s', \text{done}) )$ are stored in a replay buffer
* Mini-batches are sampled uniformly at random
* Leads to more stable and sample-efficient learning

---

### Exploration Strategy (ε-Greedy)

Actions are selected according to:

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \varepsilon \\
\arg\max_a Q(s,a), & \text{with probability } 1 - \varepsilon
\end{cases}
$$

#### Warm-up Strategy

* First **1000 episodes** use ( $\varepsilon = 1.0$ ) (pure exploration)
* After warm-up, ε decays exponentially to a minimum value

This explains the **initial negative reward plateau** observed in training logs.

---

### Training Behavior

* Rewards steadily increase after warm-up
* Agent consistently achieves **20–30 food items**
* Learned policy transfers reliably to the real game environment
* No environment mismatch after final fixes

---

## Hyperparameters

| Parameter              | Value    | Description                        |
| ---------------------- | -------- | ---------------------------------- |
| State dimension        | 11       | Compact directional representation |
| Action dimension       | 3        | Relative actions                   |
| Hidden layers          | 2        | Fully connected                    |
| Hidden units           | 256      | Per hidden layer                   |
| Discount factor (γ)    | 0.99     | Long-term reward emphasis          |
| Learning rate          | 1e-3     | Adam optimizer                     |
| Replay buffer size     | 1500     | Experience memory                  |
| Batch size             | 64       | Mini-batch size                    |
| Target update interval | Periodic | Stabilizes training                |
| Epsilon start          | 1.0      | Full exploration                   |
| Epsilon decay start    | 0.2      | Post warm-up                       |
| Epsilon minimum        | 0.01     | Persistent exploration             |
| Warm-up episodes       | 1000     | Exploration-only phase             |
| Training episodes      | 5k–20k   | Extended training improves policy  |

---

## Results

* Learned policy significantly outperforms random and naive heuristics
* Successfully navigates **dynamic obstacles** (snake body)
* Reward shaping and hyperparameter tuning further improve performance
* Enables meaningful comparison between **learning-based** and **classical planning** methods

---


Just tell me.
