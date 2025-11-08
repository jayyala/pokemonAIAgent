"""Abstract agent interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Agent(ABC):
    """Common interface for all agent implementations.

    Agents receive an observation vector and must return an action index.
    Agents that learn may also expose ``update()`` to consume experience.
    """

    @abstractmethod
    def select_action(self, obs: np.ndarray) -> int:
        """Choose an action given the current observation.

        Args:
            obs: Float32 feature vector produced by ``GameState.as_vector()``.

        Returns:
            Integer action index in ``[0, N_ACTIONS)``.
        """

    def update(self, *args: object, **kwargs: object) -> dict[str, float]:
        """Optional learning step.

        Returns:
            Dictionary of scalar metrics (e.g. ``{"loss": 0.42}``).
            Non-learning agents may return an empty dict.
        """
        return {}

    def save(self, path: str) -> None:
        """Persist agent weights/state to *path*."""

    def load(self, path: str) -> None:
        """Restore agent weights/state from *path*."""
