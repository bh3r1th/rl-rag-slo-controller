import os
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


class OpenAILLMClient(BaseLLMClient):
    """
    OpenAI-based LLM client implementation for the RAG controller.

    This implementation:
    - Uses the OpenAI chat completions API.
    - Reads the API key from the OPENAI_API_KEY environment variable.
    - Supports a simple mapping from model_size ("small", "base") to model names.
    - Estimates token counts by splitting on whitespace (approximate only).
    """

    def __init__(self, small_model: str = "gpt-4o-mini", base_model: str = "gpt-4o"):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "OpenAILLMClient requires the official OpenAI Python client."
            ) from exc

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable must be set for OpenAILLMClient."
            )

        self._client_class = OpenAI
        self._api_key = api_key
        self._small_model = small_model
        self._base_model = base_model

    def _select_model(self, model_size: str) -> str:
        size = (model_size or "").lower()
        if size == "small":
            return self._small_model
        # default to base
        return self._base_model

    def answer_with_context(
        self,
        question: str,
        docs: List[Dict[str, Any]],
        model_size: str,
        guarded: bool,
    ) -> Tuple[str, int, int]:
        """
        Call the OpenAI chat completion API with a simple RAG-style prompt.

        - Concatenates documents into a context section.
        - Optionally instructs the model to refuse if it cannot answer from the context when guarded=True.
        """
        # Instantiate client per call to avoid global state issues
        client = self._client_class(api_key=self._api_key)

        context_parts = [d.get("text", "") for d in docs]
        context = "\n\n".join(context_parts)

        system_instructions = [
            "You are a helpful assistant answering questions based ONLY on the provided context.",
        ]
        if guarded:
            system_instructions.append(
                "If the answer is not clearly supported by the context, explicitly say you cannot safely answer."
            )

        system_prompt = " ".join(system_instructions)

        model_name = self._select_model(model_size)

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{question}",
            },
        ]

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
        )

        answer = response.choices[0].message.content or ""
        n_tokens_generated = len(answer.split())
        n_tokens_context = len(context.split())
        return answer, n_tokens_generated, n_tokens_context
