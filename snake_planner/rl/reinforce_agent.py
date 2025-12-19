import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.net(x)


class ReinforceAgent:
    def __init__(
        self, state_dim: int, action_dim: int, lr: float = 5e-4, gamma: float = 0.99
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma

        # Memory for batch updates
        self.batch_log_probs = []
        self.batch_rewards = []
        self.batch_entropies = []

    def act(self, state: np.ndarray):
        state_t = torch.FloatTensor(state).to(self.device)
        probs = self.policy_net(state_t)

        # Handle potential NaNs by ensuring probs aren't zero
        probs = torch.clamp(probs, min=1e-8)

        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action), m.entropy()

    def store_outcome(self, log_prob, reward, entropy):
        self.batch_log_probs.append(log_prob)
        self.batch_rewards.append(reward)
        self.batch_entropies.append(entropy)

    def update(self):
        """Perform Policy Gradient Update with Entropy Bonus."""
        if not self.batch_rewards:
            return

        # Calculate discounted returns
        R = 0
        returns = []
        for r in reversed(self.batch_rewards):
            if (
                r <= -10
            ):  # Reset return on death to avoid bleeding death penalty backwards too far
                R = r
            else:
                R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.FloatTensor(returns).to(self.device)

        # Standardize returns for stability
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        policy_loss = []
        entropy_coeff = 0.01

        for log_prob, Gt, entropy in zip(
            self.batch_log_probs, returns, self.batch_entropies
        ):
            # We want to maximize: (log_prob * Gt) + (entropy_coeff * entropy)
            # So we minimize the negative
            policy_loss.append(-log_prob * Gt - entropy_coeff * entropy)

        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).mean()
        loss.backward()

        # Clip gradients to prevent "exploding gradients" which causes poor performance
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Clear memory for next batch
        self.batch_log_probs = []
        self.batch_rewards = []
        self.batch_entropies = []

    def save(self, path: str):
        """Save the policy network weights."""
        torch.save(self.policy_net.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load the policy network weights."""
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.policy_net.eval()
        print(f"Model loaded from {path}")
