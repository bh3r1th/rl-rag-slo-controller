from dataclasses import dataclass
from typing import List, Dict, Any
import json


@dataclass
class HotpotExample:
    """
    A single QA example from HotpotQA.

    context:
        Concatenated paragraphs from the documents referenced in 'supporting_facts'.
        If supporting_facts cannot be matched, fall back to concatenating all context paragraphs.
    """

    id: str
    question: str
    answer_text: str
    supporting_titles: List[str]
    context: str


def load_hotpot_qa(path: str) -> List[HotpotExample]:
    """
    Load a HotpotQA JSON file (train or dev) and return a list of HotpotExample.

    Assumes each entry has at least:
      - "qid": str
      - "question": str
      - "answer": str
      - "supporting_facts": List[[title, sentence_index], ...]
      - "context": List[[title, [sent_0, sent_1, ...]], ...]

    For context:
      - Collect all paragraphs from documents whose titles appear in supporting_facts.
      - Join them with newlines.
      - If supporting_facts is empty or no titles match, fall back to joining ALL context paragraphs.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples: List[HotpotExample] = []

    for item in data:
        qid = str(item.get("qid", ""))
        question = str(item.get("question", ""))
        answer = str(item.get("answer", ""))

        supporting_facts = item.get("supporting_facts", []) or []
        supporting_titles = list(
            {sf[0] for sf in supporting_facts if isinstance(sf, list) and len(sf) >= 1}
        )

        context_entries = item.get("context", []) or []
        title_to_paragraph: Dict[str, str] = {}
        all_paragraphs: List[str] = []

        for ctx in context_entries:
            if not isinstance(ctx, list) or len(ctx) != 2:
                continue
            title, sentences = ctx
            if not isinstance(title, str):
                continue
            if not isinstance(sentences, list):
                continue
            paragraph = " ".join(str(s) for s in sentences)
            all_paragraphs.append(paragraph)
            title_to_paragraph[title] = paragraph

        chosen_paragraphs: List[str] = []
        for t in supporting_titles:
            if t in title_to_paragraph:
                chosen_paragraphs.append(title_to_paragraph[t])

        if not chosen_paragraphs:
            chosen_paragraphs = all_paragraphs

        context = "\n\n".join(chosen_paragraphs)

        if not qid or not question or context is None:
            continue

        examples.append(
            HotpotExample(
                id=qid,
                question=question,
                answer_text=answer,
                supporting_titles=supporting_titles,
                context=context,
            )
        )

    return examples


def build_hotpot_corpus(examples: List[HotpotExample]) -> List[Dict[str, str]]:
    """
    Build a simple corpus list from HotpotQA examples.

    Returns a list of dicts:
      { "id": str, "text": str }

    Uses example.id and example.context directly.
    """
    corpus: List[Dict[str, str]] = []
    for ex in examples:
        corpus.append({"id": ex.id, "text": ex.context})
    return corpus
