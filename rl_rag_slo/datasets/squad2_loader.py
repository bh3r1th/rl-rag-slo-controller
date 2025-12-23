from dataclasses import dataclass
from typing import List, Optional, Dict
import json


@dataclass
class QAExample:
    """
    A single QA example from SQuAD 2.0.
    For unanswerable questions, answer_text is None.
    """

    id: str
    question: str
    context: str
    answer_text: Optional[str]


def load_squad2_qa(path: str) -> List[QAExample]:
    """
    Load a SQuAD 2.0 JSON file and return a flat list of QAExample.

    Assumes the standard SQuAD v2.0 format:

    {
      "data": [
        {
          "title": ...,
          "paragraphs": [
            {
              "context": "...",
              "qas": [
                {
                  "id": "...",
                  "question": "...",
                  "is_impossible": bool,
                  "answers": [{ "text": "...", ... }, ...]
                },
                ...
              ]
            },
            ...
          ]
        },
        ...
      ]
    }

    For answerable questions:
      - use the first answer's "text" as answer_text.
    For unanswerable questions:
      - set answer_text = None.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples: List[QAExample] = []

    for article in data.get("data", []):
        for para in article.get("paragraphs", []):
            context = para.get("context", "")
            for qa in para.get("qas", []):
                qid = qa.get("id", "")
                question = qa.get("question", "")
                is_impossible = qa.get("is_impossible", False)
                answers = qa.get("answers", []) or []

                if is_impossible or len(answers) == 0:
                    answer_text: Optional[str] = None
                else:
                    answer_text = str(answers[0].get("text", "") or "")

                if not qid or not question or context is None:
                    continue

                examples.append(
                    QAExample(
                        id=str(qid),
                        question=str(question),
                        context=str(context),
                        answer_text=answer_text
                        if answer_text
                        else (None if is_impossible else ""),
                    )
                )

    return examples


def build_squad2_corpus(examples: List[QAExample]) -> List[Dict[str, str]]:
    """
    Build a simple corpus list from SQuAD 2.0 examples.

    Returns a list of dicts:
      { "id": str, "text": str }

    Strategy:
    - Use the context string itself as the key for deduplication.
    - Assign synthetic ids "ctx_<index>".
    """
    corpus: List[Dict[str, str]] = []
    seen_contexts: Dict[str, str] = {}
    for ctx_index, example in enumerate(examples):
        ctx = example.context
        if ctx in seen_contexts:
            continue
        doc_id = f"ctx_{ctx_index}"
        seen_contexts[ctx] = doc_id
        corpus.append({"id": doc_id, "text": ctx})
    return corpus
