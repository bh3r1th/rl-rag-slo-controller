from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .policy_network import PolicyNetwork


@dataclass
class BanditBatch:
    """
    Offline replay buffer batch for contextual bandit training.
    """

    states: torch.Tensor  # (N, state_dim), float32
    actions: torch.Tensor  # (N,), int64
    rewards: torch.Tensor  # (N,), float32


class BanditTrainer:
    """
    Trainer for an offline contextual bandit using PolicyNetwork.
    """

    def __init__(
        self, state_dim: int, num_actions: int, lr: float = 1e-3, device: str = "cpu"
    ) -> None:
        self.device = torch.device(device)
        self.policy = PolicyNetwork(state_dim, num_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def _make_loader(self, replay: BanditBatch, batch_size: int) -> DataLoader:
        ds = TensorDataset(
            replay.states.to(self.device),
            replay.actions.to(self.device),
            replay.rewards.to(self.device),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=True)

    def train_epoch(self, replay: BanditBatch, batch_size: int = 128) -> float:
        """
        Run one training epoch over the replay buffer and return average loss.
        Uses a simple REINFORCE-style objective on logged data:
            loss = -E[ log pi(a|s) * reward ]
        """
        loader = self._make_loader(replay, batch_size)
        total_loss = 0.0
        total_n = 0

        for states, actions, rewards in loader:
            logits = self.policy(states)
            log_probs = F.log_softmax(logits, dim=-1)
            chosen_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            loss = -(chosen_log_probs * rewards).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_size_actual = states.size(0)
            total_loss += loss.item() * batch_size_actual
            total_n += batch_size_actual

        if total_n == 0:
            return 0.0
        return total_loss / total_n

    def save(self, path: str) -> None:
        """
        Save policy state_dict to the given path.
        """
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str) -> None:
        """
        Load policy state_dict from the given path.
        """
        state_dict = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(state_dict)
