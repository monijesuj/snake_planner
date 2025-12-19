import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from ..config import GameConfig
from .snake_env_rl import SnakeRLEnv
from .dqn_agent import DQNAgent
from .reinforce_agent import ReinforceAgent


def train_reinforce(episodes: int = 5000, batch_size: int = 10):
    config = GameConfig(grid_size=20, speed=0)
    env = SnakeRLEnv(config)
    agent = ReinforceAgent(state_dim=11, action_dim=3)

    episode_rewards = []

    # Progress bar
    pbar = tqdm(range(episodes), desc="Training REINFORCE")

    for episode in pbar:
        state = env.reset()
        done = False
        total_reward = 0

        # Collect one episode
        while not done:
            action, log_prob, entropy = agent.act(state)
            next_state, reward, done = env.step(action)

            agent.store_outcome(log_prob, reward, entropy)
            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)

        # Update policy after batch_size episodes
        if (episode + 1) % batch_size == 0:
            agent.update()

        if (episode + 1) % 100 == 0:
            avg = np.mean(episode_rewards[-100:])
            pbar.set_postfix({"Avg Reward (100)": f"{avg:.2f}"})

    agent.save("reinforce_snake.pt")
    plot_rewards(episode_rewards, window=100, title="REINFORCE Training Performance")


def train_dqn(episodes: int = 5000, target_update: int = 10):
    config = GameConfig(grid_size=20, speed=0)
    env = SnakeRLEnv(config)
    agent = DQNAgent(state_dim=11, action_dim=3)

    epsilon_start, epsilon_min = 0.2, 0.01
    warmup_episodes = 1000
    decay_rate = np.log(epsilon_start / epsilon_min) / (episodes - warmup_episodes)
    episode_rewards = []

    pbar = tqdm(range(episodes), desc="Training DQN")
    for episode in pbar:
        state = env.reset()
        total_reward, done = 0.0, False

        epsilon = (
            1.0
            if episode < warmup_episodes
            else max(
                epsilon_min,
                epsilon_start * np.exp(-decay_rate * (episode - warmup_episodes)),
            )
        )

        while not done:
            action = agent.act(state, epsilon)
            next_state, reward, done = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.update()
            state, total_reward = next_state, total_reward + reward

        episode_rewards.append(total_reward)
        if episode % target_update == 0:
            agent.update_target()

        if episode % 10 == 0:
            pbar.set_postfix({"Reward": f"{total_reward:.1f}", "Eps": f"{epsilon:.2f}"})

    torch.save(agent.q_net.state_dict(), "dqn_snake.pt")
    plot_rewards(episode_rewards, window=100, title="DQN Training Performance")


def plot_rewards(rewards, window=50, title="Training Performance"):
    """
    Plot raw and smoothed reward curves with dynamic title.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, color="blue", alpha=0.15, label="Episode Reward")

    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        plt.plot(
            range(window - 1, len(rewards)),
            smoothed,
            color="red",
            linewidth=2,
            label=f"Moving Average ({window})",
        )

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()


if __name__ == "__main__":
    # To train REINFORCE:
    train_reinforce(episodes=8000)
    # To train DQN:
    # train_dqn(episodes=5000)
