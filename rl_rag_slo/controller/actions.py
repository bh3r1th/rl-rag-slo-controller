from dataclasses import dataclass
from typing import Dict, Literal


@dataclass(frozen=True)
class RagActionConfig:
    """
    Configuration for a discrete RAG controller action.

    Fields:
        action_id: Integer identifier for the action.
        k: Retrieval depth (top-k documents to fetch).
        answer_mode: Behavior of the answer generator:
            - "guarded": allowed to refuse if unsure.
            - "auto": always attempts an answer.
            - "refuse": explicit refusal (no retrieval, no LLM call).
    """
    action_id: int
    k: int
    answer_mode: Literal["auto", "guarded", "refuse"]


# Simplified, interpretable 5-action space.
# Model is always gpt-4.1-nano (enforced by OpenAILLMClient defaults),
# so we don't vary model_size here.

ACTIONS: Dict[int, RagActionConfig] = {
    # Cheaper / lower recall
    0: RagActionConfig(
        action_id=0,
        k=2,
        answer_mode="guarded",
    ),

    # Midpoint baseline
    1: RagActionConfig(
        action_id=1,
        k=5,
        answer_mode="guarded",
    ),

    # High recall / higher cost
    2: RagActionConfig(
        action_id=2,
        k=10,
        answer_mode="guarded",
    ),

    # Aggressive: same k=5 but "auto" (no safety/refusal instructions)
    3: RagActionConfig(
        action_id=3,
        k=5,
        answer_mode="auto",
    ),

    # Conservative: explicit refusal (no retrieval, no LLM call)
    4: RagActionConfig(
        action_id=4,
        k=0,
        answer_mode="refuse",
    ),
}
