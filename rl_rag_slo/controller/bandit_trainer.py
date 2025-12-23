from dataclasses import dataclass
from typing import Dict, List, Optional

from .actions import RagActionConfig


@dataclass
class BanditBatch:
    """Batch of bandit feedback data."""

    rewards: List[float]
    actions: List[int]
    metadata: Optional[Dict[str, str]] = None


class BanditTrainer:
    """Skeleton trainer for bandit-style updates."""

    def __init__(self, action_space: List[RagActionConfig]) -> None:
        """Initialize the trainer with available actions.

        TODO: set up optimization state and metrics.
        """
        self._action_space = action_space

    def record_feedback(self, reward: float, action_id: int) -> None:
        """Record reward feedback for a taken action.

        TODO: store feedback in a replay buffer.
        """
        # TODO: implement feedback storage.
        return None

    def train_step(self, batch: BanditBatch) -> None:
        """Perform a single training step.

        TODO: implement bandit update logic.
        """
        # TODO: update policy based on bandit feedback.
        return None
