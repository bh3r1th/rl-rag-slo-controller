from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class SloProfile:
    """Represent a latency/quality SLO profile for the controller."""

    name: str
    targets: Dict[str, float]


def list_default_profiles() -> List[SloProfile]:
    """Return default SLO profiles.

    TODO: Populate profiles from configuration or constants.
    """
    # TODO: build out concrete profile definitions.
    return []
