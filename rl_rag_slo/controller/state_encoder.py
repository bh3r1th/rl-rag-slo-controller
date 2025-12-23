from dataclasses import dataclass
from typing import Dict, List


@dataclass
class EncodedState:
    """Container for encoded controller state features."""

    vector: List[float]
    metadata: Dict[str, str]


class StateEncoder:
    """Encode raw controller observations into model-ready features."""

    def __init__(self, feature_names: List[str]) -> None:
        """Initialize the encoder.

        TODO: validate feature names and configure normalization.
        """
        self._feature_names = feature_names

    def encode(self, observation: Dict[str, float]) -> EncodedState:
        """Encode a raw observation into a feature vector.

        TODO: implement feature extraction and normalization.
        """
        # TODO: build encoded vector based on configured feature names.
        return EncodedState(vector=[], metadata={})
