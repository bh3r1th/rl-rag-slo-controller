from typing import Callable, Dict, Any, Optional

import numpy as np


class StateEncoder:
    """
    Encodes (question, domain_id, SLO vector, extra metadata) into a flat NumPy state vector.
    """

    def __init__(self, embedder: Callable[[str], np.ndarray], num_domains: int = 4):
        """
        embedder: callable mapping question text -> embedding vector (np.ndarray)
        num_domains: number of domains for one-hot encoding.
        """
        self.embedder = embedder
        self.num_domains = num_domains

    def encode(
        self,
        question: str,
        domain_id: int,
        slo_vec: np.ndarray,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Build the state vector as concatenation of:
        - question embedding
        - one-hot domain_id of length self.num_domains
        - SLO vector (float32)
        - extra features from extra_meta
        Returns np.ndarray with dtype float32.
        """
        q_emb = self.embedder(question).astype(np.float32)
        domain_one_hot = self._domain_one_hot(domain_id)
        slo_scaled = slo_vec.astype(np.float32) * 5.0

        meta = extra_meta or {}
        q_len_tokens = int(meta.get("q_len_tokens", 0) or 0)
        wh_one_hot = meta.get("wh_one_hot", np.zeros(7, dtype=np.float32))
        bm25_max = float(meta.get("bm25_max", 0.0) or 0.0)
        bm25_mean = float(meta.get("bm25_mean", 0.0) or 0.0)
        bm25_gap = float(meta.get("bm25_gap", 0.0) or 0.0)

        q_len_scaled = float(q_len_tokens) / 100.0
        wh_vec = np.asarray(wh_one_hot, dtype=np.float32)
        bm_vec = np.array([bm25_max, bm25_mean, bm25_gap], dtype=np.float32)

        extra_features = np.concatenate(
            [
                np.array([q_len_scaled], dtype=np.float32),
                wh_vec.astype(np.float32),
                bm_vec,
            ],
            axis=0,
        )

        state = np.concatenate(
            [q_emb, domain_one_hot, slo_scaled, extra_features], axis=0
        ).astype(np.float32)
        return state

    def _infer_q_type(self, q: str) -> np.ndarray:
        """
        Infer question type from prefix and return 8D one-hot:
        [what, why, how, when, where, who, yesno, other].
        """
        types = ["what", "why", "how", "when", "where", "who"]
        q_lower = q.strip().lower()
        one_hot = np.zeros(8, dtype=np.float32)
        for i, t in enumerate(types):
            if q_lower.startswith(t):
                one_hot[i] = 1.0
                return one_hot
        yesno_prefixes = ["is", "are", "do", "does", "did", "can"]
        if any(q_lower.startswith(p) for p in yesno_prefixes):
            one_hot[6] = 1.0  # yesno
        else:
            one_hot[7] = 1.0  # other
        return one_hot

    def _domain_one_hot(self, domain_id: int) -> np.ndarray:
        """
        One-hot encode domain_id into a vector of length self.num_domains.
        Clamp domain_id to [0, num_domains-1].
        """
        idx = max(0, min(domain_id, self.num_domains - 1))
        vec = np.zeros(self.num_domains, dtype=np.float32)
        vec[idx] = 1.0
        return vec

    def _encode_extra(self, extra_meta: Dict[str, Any]) -> np.ndarray:
        """
        Encode optional metadata into extra numeric features.
        Known keys:
          - 'avg_score'
          - 'max_score'
          - 'num_candidates'
        Missing keys are skipped. Returns float32 vector (possibly empty).
        """
        features = []
        for key in ("avg_score", "max_score", "num_candidates"):
            if key in extra_meta and extra_meta[key] is not None:
                try:
                    features.append(float(extra_meta[key]))
                except (TypeError, ValueError):
                    continue
        if not features:
            return np.zeros(0, dtype=np.float32)
        return np.array(features, dtype=np.float32)
