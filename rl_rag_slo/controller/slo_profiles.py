"""SLO profile vectors used to weight quality, cost, refusal, and hallucination."""

from typing import Dict

import numpy as np

# Each vector: [w_quality, w_cost, w_refusal, w_halluc]
SLO_PROFILES: Dict[str, np.ndarray] = {
    "balanced": np.array([0.5, 0.2, 0.1, 0.2], dtype=np.float32),
    "cheap": np.array([0.4, 0.4, 0.1, 0.1], dtype=np.float32),
    "safe": np.array([0.5, 0.1, 0.1, 0.3], dtype=np.float32),
}


def get_slo_vector(name: str) -> np.ndarray:
    """
    Return a COPY of the SLO vector for the given profile name as float32.
    Raise KeyError if the name is unknown.
    """
    vec = SLO_PROFILES[name]
    return vec.astype(np.float32).copy()
