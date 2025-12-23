"""SLO profile definitions for the RAG controller."""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class SLOProfile:
    """Represents target service-level objectives for the controller."""

    name: str
    targets: Dict[str, float]
    weights: Optional[Dict[str, float]] = None


class SLOProfileStore:
    """Container for managing available SLO profiles."""

    def __init__(self) -> None:
        """Initialize an empty store."""
        self._profiles: Dict[str, SLOProfile] = {}

    def add(self, profile: SLOProfile) -> None:
        """Add a new profile to the store.

        TODO: validate profile contents.
        """
        # TODO: Validate targets/weights ranges.
        self._profiles[profile.name] = profile

    def get(self, name: str) -> SLOProfile:
        """Fetch a profile by name.

        TODO: handle missing profiles gracefully.
        """
        # TODO: Add missing-profile handling.
        return self._profiles[name]

    def list_profiles(self) -> Dict[str, SLOProfile]:
        """Return all stored profiles.

        TODO: consider returning a copy to prevent mutation.
        """
        # TODO: Return a copy if needed.
        return self._profiles
