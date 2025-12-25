from typing import Dict, Any

import numpy as np


def classify_wh_type(question: str) -> str:
    q = (question or "").lstrip().lower()
    if q.startswith("who"):
        return "who"
    if q.startswith("what"):
        return "what"
    if q.startswith("when"):
        return "when"
    if q.startswith("where"):
        return "where"
    if q.startswith("why"):
        return "why"
    if q.startswith("how"):
        return "how"
    return "other"


def wh_one_hot(wh_type: str) -> np.ndarray:
    order = ["who", "what", "when", "where", "why", "how", "other"]
    vec = np.zeros(len(order), dtype=np.float32)
    t = wh_type if wh_type in order else "other"
    vec[order.index(t)] = 1.0
    return vec


def compute_question_features(question: str) -> Dict[str, Any]:
    tokens = (question or "").split()
    wh_type = classify_wh_type(question)
    return {
        "q_len_tokens": int(len(tokens)),
        "wh_type": wh_type,
        "wh_one_hot": wh_one_hot(wh_type),
    }


def compute_bm25_features(
    retriever, question: str, top_k: int = 5
) -> Dict[str, float]:
    results = retriever.retrieve(question, top_k=top_k) if retriever else []
    scores = [float(r.get("score", 0.0)) for r in results]

    if not scores:
        return {
            "bm25_max": 0.0,
            "bm25_mean": 0.0,
            "bm25_gap": 0.0,
        }

    max_score = float(np.max(scores))
    mean_score = float(np.mean(scores))
    gap = float(scores[0] - scores[1]) if len(scores) >= 2 else 0.0

    return {
        "bm25_max": max_score,
        "bm25_mean": mean_score,
        "bm25_gap": gap,
    }
