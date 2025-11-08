"""Hand-crafted heuristic agent for Pokémon Red.

The heuristic uses interpretable rules derived from game knowledge:

* **Battle mode** – mash A (attack) most of the time; occasionally press B
  to go back or try a different move.
* **Overworld** – randomly choose a cardinal direction; occasionally
  press A to interact with NPCs / advance dialog.

This agent serves as a competitive baseline that requires no training and
is deterministic given a fixed seed.
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np

from ..environment.actions import Action, BATTLE_ACTIONS, MOVEMENT_ACTIONS
from .base import Agent


class HeuristicAgent(Agent):
    """Rule-based agent using hand-coded game knowledge.

    Args:
        battle_a_prob:    Probability of pressing A during a battle turn.
        confirm_prob:     Probability of pressing A in overworld (dialog / interact).
        seed:             RNG seed for reproducibility.
    """

    def __init__(
        self,
        battle_a_prob: float = 0.80,
        confirm_prob: float = 0.25,
        seed: Optional[int] = None,
    ) -> None:
        self._battle_a_prob = battle_a_prob
        self._confirm_prob  = confirm_prob
        self._rng           = random.Random(seed)

    def select_action(self, obs: np.ndarray) -> int:
        """Pick an action based on heuristic rules.

        The 4th element of the observation vector (index 3) encodes the
        battle_type normalised to [0, 1].  Values > 0 indicate an active
        battle.
        """
        in_battle: bool = obs[3] > 0.0

        if in_battle:
            return self._battle_action()
        return self._overworld_action()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _battle_action(self) -> int:
        r = self._rng.random()
        if r < self._battle_a_prob:
            return Action.A
        if r < self._battle_a_prob + 0.10:
            return Action.B
        # Navigate move menu with directional inputs.
        return self._rng.choice(BATTLE_ACTIONS)

    def _overworld_action(self) -> int:
        r = self._rng.random()
        if r < self._confirm_prob:
            return Action.A
        return self._rng.choice(MOVEMENT_ACTIONS)
