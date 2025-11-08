"""Random baseline agent.

Selects uniformly at random from the full action set.  Useful as a
lower-bound baseline and for sanity-checking the environment plumbing.
"""

from __future__ import annotations

import random

import numpy as np

from ..environment.actions import N_ACTIONS
from .base import Agent


class RandomAgent(Agent):
    """Uniformly-random action selection."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def select_action(self, obs: np.ndarray) -> int:  # noqa: ARG002
        return self._rng.randint(0, N_ACTIONS - 1)
