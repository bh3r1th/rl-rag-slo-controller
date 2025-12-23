"""Action definitions for the RAG SLO controller."""

from dataclasses import dataclass
from typing import Dict, Literal


@dataclass(frozen=True)
class RagActionConfig:
    """
    Configuration for a discrete RAG controller action.
    """

    action_id: int
    k: int  # retrieval depth
    model_size: Literal["small", "base"]
    answer_mode: Literal["auto", "guarded", "refuse"]


class ActionRegistry:
    """Registry for mapping action IDs to RAG action configurations."""

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._actions: Dict[int, RagActionConfig] = {}

    def register(self, config: RagActionConfig) -> None:
        """Register a new action configuration.

        TODO: enforce uniqueness and validation.
        """
        # TODO: Add validation and duplicate checks.
        self._actions[config.action_id] = config

    def get(self, action_id: int) -> RagActionConfig:
        """Retrieve a configuration by action ID.

        TODO: raise a custom error when missing.
        """
        # TODO: Add error handling for unknown IDs.
        return self._actions[action_id]

    def all(self) -> Dict[int, RagActionConfig]:
        """Return all registered actions.

        TODO: consider returning a defensive copy.
        """
        # TODO: Return a copy if mutation should be prevented.
        return self._actions
