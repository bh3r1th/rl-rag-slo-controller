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


def build_action_lookup(actions: Dict[int, RagActionConfig]) -> Dict[int, RagActionConfig]:
    """Return a lookup table for action configs.

    TODO: Validate the incoming action registry.
    """
    # TODO: implement validation, normalization, and defaults.
    return dict(actions)
