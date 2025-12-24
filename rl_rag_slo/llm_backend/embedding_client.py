from abc import ABC, abstractmethod
from typing import List
import os
import numpy as np


class BaseEmbedder(ABC):
    """
    Abstract base class for text embedding backends.
    """

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """
        Return a 1D NumPy embedding vector for the given text.
        """
        raise NotImplementedError


class DeterministicHashEmbedder(BaseEmbedder):
    """
    Deterministic pseudo-random embedder based on a hash of the input text.
    Useful for testing the RL pipeline without calling external APIs.
    """

    def __init__(self, dim: int = 128):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        """
        Return a deterministic pseudo-random embedding based on a hash of the input text.

        We clamp the seed into the 32-bit range accepted by numpy.RandomState
        to avoid ValueError: "Seed must be between 0 and 2**32 - 1".
        """
        import hashlib

        h = hashlib.sha256(text.encode("utf-8")).digest()
        # Take 8 bytes, interpret as integer, then clamp into [0, 2**32 - 1]
        raw_seed = int.from_bytes(h[:8], "little", signed=False)
        seed = raw_seed % (2**32 - 1)
        rng = np.random.RandomState(seed)
        return rng.normal(loc=0.0, scale=1.0, size=(self.dim,)).astype(np.float32)


class OpenAIEmbedder(BaseEmbedder):
    """
    OpenAI embedding client.

    Uses the OpenAI embeddings API, reading the key from OPENAI_API_KEY.
    """

    def __init__(self, model: str = "text-embedding-3-small"):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise ImportError("OpenAIEmbedder requires the official OpenAI Python client.") from exc

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable must be set for OpenAIEmbedder."
            )

        self._client_class = OpenAI
        self._api_key = api_key
        self._model = model

    def embed(self, text: str) -> np.ndarray:
        """
        Call the OpenAI embeddings API and return the vector as float32 NumPy array.
        """
        client = self._client_class(api_key=self._api_key)
        resp = client.embeddings.create(
            model=self._model,
            input=text,
        )
        vec = resp.data[0].embedding
        return np.asarray(vec, dtype=np.float32)
