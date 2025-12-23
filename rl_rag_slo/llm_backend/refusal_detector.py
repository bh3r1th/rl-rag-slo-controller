from typing import List


def is_refusal(answer: str) -> bool:
    """
    Return True if the answer looks like a refusal.

    This uses a simple keyword-based heuristic with case-insensitive matching.
    It checks for phrases such as:
    - "cannot answer"
    - "cannot safely answer"
    - "do not have enough information"
    - "don't have enough information"
    - "no sufficient evidence"
    - "insufficient information"
    - "cannot determine"
    - "unable to determine"
    """
    if not answer:
        return False

    text = answer.lower()
    patterns: List[str] = [
        "cannot answer",
        "cannot safely answer",
        "do not have enough information",
        "don't have enough information",
        "no sufficient evidence",
        "insufficient information",
        "cannot determine",
        "unable to determine",
    ]
    return any(p in text for p in patterns)
