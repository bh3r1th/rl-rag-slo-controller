from typing import Dict, Optional, Any
import re
from .refusal_detector import is_refusal


def normalize_text(text: str) -> str:
    """
    Normalize text for simple exact-match style scoring:
    - lowercase
    - strip leading/trailing whitespace
    - remove punctuation
    - collapse multiple whitespace characters into a single space
    """
    if text is None:
        return ""
    text = text.lower().strip()
    # remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def exact_match(prediction: str, ground_truth: str) -> float:
    """
    Return 1.0 if normalized prediction equals normalized ground_truth, else 0.0.
    """
    pred_norm = normalize_text(prediction)
    gold_norm = normalize_text(ground_truth)
    return 1.0 if pred_norm == gold_norm and gold_norm != "" else 0.0


def compute_qa_score(
    answer: str,
    ground_truth: Optional[str],
    extra_eval: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute basic QA metrics for a single example.

    Behavior:

    If ground_truth is not None (answerable question):
        - accuracy = exact_match(answer, ground_truth)
        - hallucination = 1 if answer is non-empty and accuracy == 0.0, else 0
        - correct_refusal = 0
        - wrong_refusal = 1 if is_refusal(answer) else 0

    If ground_truth is None (unanswerable question):
        - accuracy = 0.0
        - hallucination = 1 if answer is non-empty and not is_refusal(answer), else 0
        - correct_refusal = 1 if is_refusal(answer) else 0
        - wrong_refusal = 0

    The result dictionary must contain at least:
      - "accuracy": float
      - "hallucination": int
      - "correct_refusal": int
      - "wrong_refusal": int

    If extra_eval is provided, its key/value pairs are merged into the result.
    """
    result: Dict[str, Any] = {
        "accuracy": 0.0,
        "hallucination": 0,
        "correct_refusal": 0,
        "wrong_refusal": 0,
    }

    ans_str = answer or ""
    pred_is_refusal = is_refusal(ans_str)
    has_answer_text = ans_str.strip() != ""

    if ground_truth is not None:
        # Answerable
        acc = 0.0
        if ground_truth:
            gt = " ".join(ground_truth.strip().lower().split())
            ans = " ".join(str(answer).strip().lower().split())
            if gt and gt in ans:
                acc = 1.0
        hallucination = 1 if has_answer_text and acc == 0.0 else 0
        correct_refusal = 0
        wrong_refusal = 1 if pred_is_refusal else 0
        result.update(
            accuracy=float(acc),
            hallucination=int(hallucination),
            correct_refusal=int(correct_refusal),
            wrong_refusal=int(wrong_refusal),
        )
    else:
        # Unanswerable
        accuracy = 0.0
        hallucination = 1 if has_answer_text and not pred_is_refusal else 0
        correct_refusal = 1 if pred_is_refusal else 0
        wrong_refusal = 0
        result.update(
            accuracy=float(accuracy),
            hallucination=int(hallucination),
            correct_refusal=int(correct_refusal),
            wrong_refusal=int(wrong_refusal),
        )

    if extra_eval:
        result.update(extra_eval)

    return result
