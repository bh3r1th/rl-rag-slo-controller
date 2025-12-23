"""Bandit training scaffolding for the RAG controller."""

from dataclasses import dataclass
from typing import Dict, List

from rl_rag_slo.controller.actions import RagActionConfig
from rl_rag_slo.controller.state_encoder import StateEncoding


@dataclass
class BanditStep:
    """Record of a single bandit interaction."""

    state: StateEncoding
    action: RagActionConfig
    reward: float
    metadata: Dict[str, float]


class BanditTrainer:
    """Placeholder trainer for contextual bandit updates."""

    def __init__(self) -> None:
        """Initialize training buffers and hyperparameters.

        TODO: accept training configuration.
        """
        self._history: List[BanditStep] = []

    def record_step(self, step: BanditStep) -> None:
        """Record a bandit step for offline training.

        TODO: apply retention or sampling strategies.
        """
        # TODO: Add replay buffer logic.
        self._history.append(step)

    def train(self) -> None:
        """Run a training pass over recorded steps.

        TODO: implement optimization updates.
        """
        # TODO: Perform model updates.
        return None

    def history(self) -> List[BanditStep]:
        """Return recorded training steps.

        TODO: consider returning a copy for safety.
        """
        # TODO: Return a copy if needed.
        return list(self._history)
