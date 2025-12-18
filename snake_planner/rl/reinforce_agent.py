import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import List

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

class ReinforceAgent:
    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-3, gamma: float = 0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        
        # Episode memory
        self.log_probs = []
        self.rewards = []

    def act(self, state: np.ndarray) -> tuple:
        state_t = torch.FloatTensor(state).to(self.device)
        probs = self.policy_net(state_t)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def store_outcome(self, log_prob, reward):
        self.log_probs.append(log_prob)
        self.rewards.append(reward)

    def update(self):
        """Perform Policy Gradient Update."""
        R = 0
        policy_loss = []
        returns = []
        
        # Calculate discounted returns (G_t)
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
            
        returns = torch.FloatTensor(returns).to(self.device)
        # Optional: Standardize returns to reduce variance
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        for log_prob, Gt in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * Gt)

        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()
        
        # Clear memory
        self.log_probs = []
        self.rewards = []

    def save(self, path: str):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.policy_net.eval()