from typing import Dict
import numpy as np

"""
SLO profiles for the RL RAG controller.

Each vector is:
    [w_quality, w_cost, w_refusal, w_halluc]
"""


# You can tune these later, but start with clear, distinct trade-offs.
SLO_PROFILES: Dict[str, np.ndarray] = {
    # Original "balanced" profile
    "balanced": np.array([0.5, 0.2, 0.1, 0.2], dtype=np.float32),

    # Strongly prioritize retrieval/answer quality over cost.
    # This should push the policy toward higher-k / stronger configs
    # if they improve retrieval_hit.
    "quality_first": np.array([0.8, 0.05, 0.05, 0.10], dtype=np.float32),

    # Emphasize cheapness; expect lower hit-rate but lower cost.
    "cheap": np.array([0.3, 0.5, 0.05, 0.15], dtype=np.float32),

    # Safety / hallucination-sensitive profile (more useful once using real LLM).
    "safe": np.array([0.5, 0.1, 0.1, 0.3], dtype=np.float32),
}


def get_slo_vector(name: str) -> np.ndarray:
    """
    Return a COPY of the SLO vector for the given profile name as float32.
    Raise KeyError if the name is unknown.
    """
    vec = SLO_PROFILES[name]
    return vec.astype(np.float32).copy()
