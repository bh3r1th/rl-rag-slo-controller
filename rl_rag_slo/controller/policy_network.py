import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union


class PolicyNetwork(nn.Module):
    """
    Simple MLP policy network for a contextual bandit over discrete RAG actions.
    """

    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.logits = nn.Linear(hidden_dim, num_actions)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute logits over actions given a batch of states.
        state: shape (batch_size, state_dim)
        returns: shape (batch_size, num_actions)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.logits(x)

    def act(self, state: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        """
        Sample an action index for a single state or batch of states
        using softmax with the given temperature.
        If input state is 1D, returns a scalar tensor.
        If input state is 2D, returns a tensor of shape (batch_size,).
        """
        logits = self.forward(state)
        probs = F.softmax(logits / temperature, dim=-1)
        actions = torch.multinomial(probs, num_samples=1).squeeze(-1)
        if state.dim() == 1:
            return actions.squeeze(0)
        return actions
