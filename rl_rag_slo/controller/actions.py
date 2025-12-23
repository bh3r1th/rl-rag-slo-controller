from dataclasses import dataclass
from typing import Dict, Literal


@dataclass(frozen=True)
class RagActionConfig:
    """
    Configuration for a discrete RAG controller action.

    Fields:
        action_id: Integer identifier for the action.
        k: Retrieval depth (top-k documents to fetch).
        model_size: Model size hint ("small" or "base").
        answer_mode: Behavior of the answer generator:
            - "guarded": allowed to refuse if unsure.
            - "auto": always attempts an answer.
            - "refuse": explicit refusal (no retrieval, no LLM call).
    """
    action_id: int
    k: int
    model_size: Literal["small", "base"]
    answer_mode: Literal["auto", "guarded", "refuse"]


# Discrete action space for the RAG controller.
# You can tune this later for experiments, but keep the keys stable.
ACTIONS: Dict[int, RagActionConfig] = {
    # Small model, guarded, varying k
    0: RagActionConfig(action_id=0, k=2, model_size="small", answer_mode="guarded"),
    1: RagActionConfig(action_id=1, k=5, model_size="small", answer_mode="guarded"),
    2: RagActionConfig(action_id=2, k=10, model_size="small", answer_mode="guarded"),

    # Base model, guarded, varying k
    3: RagActionConfig(action_id=3, k=2, model_size="base", answer_mode="guarded"),
    4: RagActionConfig(action_id=4, k=5, model_size="base", answer_mode="guarded"),
    5: RagActionConfig(action_id=5, k=10, model_size="base", answer_mode="guarded"),

    # Base model, auto-answer (more aggressive, less safe)
    6: RagActionConfig(action_id=6, k=2, model_size="base", answer_mode="auto"),
    7: RagActionConfig(action_id=7, k=5, model_size="base", answer_mode="auto"),
    8: RagActionConfig(action_id=8, k=10, model_size="base", answer_mode="auto"),

    # Explicit refusal action (no retrieval, no generation)
    9: RagActionConfig(action_id=9, k=0, model_size="base", answer_mode="refuse"),
}