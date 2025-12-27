import re
from typing import List


def normalize_text(s: str) -> str:
    lowered = s.lower()
    cleaned = re.sub(r"[^0-9a-zA-Z]+", " ", lowered)
    return " ".join(cleaned.split()).strip()


def compute_retrieval_hit(gold_answers: list[str], retrieved_texts: list[str]) -> int:
    if not gold_answers:
        return 0
    normalized_answers = [normalize_text(ans) for ans in gold_answers]
    normalized_answers = [ans for ans in normalized_answers if ans]
    if not normalized_answers:
        return 0
    if not retrieved_texts:
        return 0
    retrieved_text = normalize_text(" ".join(retrieved_texts))
    if not retrieved_text:
        return 0
    for ans in normalized_answers:
        if len(ans) >= 3 and ans in retrieved_text:
            return 1
    return 0


def compute_retrieval_hit_rate(answerable_flags: list[bool], hits: list[int]) -> float:
    total = 0
    count = 0
    for flag, hit in zip(answerable_flags, hits):
        if flag:
            total += int(hit)
            count += 1
    if count == 0:
        return 0.0
    return total / count


if __name__ == "__main__":
    examples = [
        (["New York"], ["I visited New York City last year."]),
        (["Berlin"], ["Paris is the capital of France."]),
    ]
    for gold, retrieved in examples:
        print(f"gold={gold} retrieved={retrieved} hit={compute_retrieval_hit(gold, retrieved)}")
    print(
        "hit_rate=",
        compute_retrieval_hit_rate(
            [True, True, False], [1, 0, 1]
        ),
    )
