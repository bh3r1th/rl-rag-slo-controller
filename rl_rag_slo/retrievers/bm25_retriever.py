from typing import List, Dict, Any
import json


class BM25Retriever:
    """
    Simple BM25-style retriever over a corpus of documents.

    Corpus format: List[{"id": str, "text": str}].
    If rank_bm25 is available, use it; otherwise, use a basic TF-IDF cosine similarity as a fallback.
    """

    def __init__(self, corpus: List[Dict[str, str]]):
        """
        Initialize internal index structures from the given corpus.
        """
        self.corpus = corpus
        self.doc_ids = [d["id"] for d in corpus]
        self.texts = [d["text"] for d in corpus]

        try:
            from rank_bm25 import BM25Okapi  # type: ignore

            self._use_bm25 = True
            tokenized_corpus = [text.split() for text in self.texts]
            self._bm25 = BM25Okapi(tokenized_corpus)
            self._vectorizer = None
            self._tfidf_matrix = None
        except ImportError:
            # Fallback: lazy TF-IDF setup
            self._use_bm25 = False
            self._bm25 = None
            self._vectorizer = None
            self._tfidf_matrix = None

    def _ensure_tfidf(self) -> None:
        """
        Lazily build TF-IDF matrix if BM25 is not available.
        """
        if self._use_bm25:
            return
        if self._vectorizer is not None:
            return
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

        self._TfidfVectorizer = TfidfVectorizer
        self._cosine_similarity = cosine_similarity
        self._vectorizer = TfidfVectorizer()
        self._tfidf_matrix = self._vectorizer.fit_transform(self.texts)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top_k documents for the given query.
        Returns list of dicts: {"id": str, "text": str, "score": float}.
        """
        if not self.corpus:
            return []

        if self._use_bm25 and self._bm25 is not None:
            tokenized_query = query.split()
            scores = self._bm25.get_scores(tokenized_query)
        else:
            self._ensure_tfidf()
            if self._vectorizer is None or self._tfidf_matrix is None:
                return []
            q_vec = self._vectorizer.transform([query])
            scores = self._cosine_similarity(q_vec, self._tfidf_matrix)[0]

        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        top = indexed_scores[:top_k]

        results: List[Dict[str, Any]] = []
        for idx, score in top:
            results.append(
                {
                    "id": self.doc_ids[idx],
                    "text": self.texts[idx],
                    "score": float(score),
                }
            )
        return results

    @classmethod
    def from_jsonl(cls, path: str) -> "BM25Retriever":
        """
        Load documents from a JSONL file where each line has:
        {
          "id": "...",
          "text": "..."
        }
        """
        corpus: List[Dict[str, str]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                doc_id = str(obj.get("id", ""))
                text = str(obj.get("text", ""))
                if not doc_id or not text:
                    continue
                corpus.append({"id": doc_id, "text": text})
        return cls(corpus)
