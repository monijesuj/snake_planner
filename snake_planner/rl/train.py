import numpy as np
import torch
import matplotlib.pyplot as plt

from ..config import GameConfig
from .snake_env_rl import SnakeRLEnv
from .dqn_agent import DQNAgent



def train_dqn(
    episodes: int = 5000,
    target_update: int = 10
):
    config = GameConfig(grid_size=20, speed=0)
    env = SnakeRLEnv(config)

    state_dim = 11
    action_dim = 3

    agent = DQNAgent(state_dim, action_dim)

    epsilon_start = 0.2
    epsilon_min = 0.01
    warmup_episodes = 1000

    decay_rate = np.log(epsilon_start / epsilon_min) / (episodes - warmup_episodes)
    episode_rewards = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0.0
        done = False
        
        if episode < warmup_episodes:
            epsilon = 1.0
        else:
            epsilon = max(
                epsilon_min,
                epsilon_start * np.exp(-decay_rate * (episode - warmup_episodes))
            )

        while not done:
            action = agent.act(state, epsilon)
            next_state, reward, done = env.step(action)

            agent.replay_buffer.push(
                state, action, reward, next_state, done
            )

            agent.update()

            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)

        if episode % target_update == 0:
            agent.update_target()

        if episode % 10 == 0:
            print(
                f"Episode {episode:4d} | "
                f"Reward: {total_reward:6.2f} | "
                f"Epsilon: {epsilon:.3f}"
            )

    torch.save(agent.q_net.state_dict(), "dqn_snake.pt")
    print("Training complete.")

    # Plot rewards
    plot_rewards(episode_rewards)


def plot_rewards(rewards, window=50):
    """
    Plot raw and smoothed reward curves.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, label="Episode Reward")

    if len(rewards) >= window:
        smoothed = np.convolve(
            rewards,
            np.ones(window) / window,
            mode="valid"
        )
        plt.plot(
            range(window - 1, len(rewards)),
            smoothed,
            label=f"Moving Average ({window})"
        )

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Training Performance")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    train_dqn()
