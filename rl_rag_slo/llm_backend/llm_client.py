from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple


class BaseLLMClient(ABC):
    """
    Abstract base class for LLM backends used by the RAG controller.
    """

    @abstractmethod
    def answer_with_context(
        self,
        question: str,
        docs: List[Dict[str, Any]],
        model_size: str,
        guarded: bool,
    ) -> Tuple[str, int, int]:
        """
        Generate an answer given the question and retrieved documents.

        Args:
            question: User query.
            docs: List of retrieved documents, each with at least a "text" field.
            model_size: Model size hint ("small", "base", etc.).
            guarded: If True, backend is allowed to refuse if unsure.

        Returns:
            answer: Generated answer string.
            n_tokens_generated: Approximate number of tokens generated in the answer.
            n_tokens_context: Approximate number of tokens in the concatenated context.
        """
        raise NotImplementedError


class DummyLLMClient(BaseLLMClient):
    """
    Dummy LLM client that does not call any external API.
    It concatenates document texts and returns a truncated slice as the answer.
    """

    def answer_with_context(
        self,
        question: str,
        docs: List[Dict[str, Any]],
        model_size: str,
        guarded: bool,
    ) -> Tuple[str, int, int]:
        """
        Build a trivial answer and estimate token counts.

        The answer format is:
            "DUMMY_ANSWER: " + first 256 characters of the concatenated context.

        Token counts are approximated by splitting on whitespace.
        """
        context_texts = [d.get("text", "") for d in docs]
        context = "\n\n".join(context_texts)
        truncated_context = context[:256]
        answer = f"DUMMY_ANSWER: {truncated_context}"
        n_tokens_generated = len(answer.split())
        n_tokens_context = len(context.split())
        return answer, n_tokens_generated, n_tokens_context
