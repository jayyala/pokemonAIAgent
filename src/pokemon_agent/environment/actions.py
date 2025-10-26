"""Action space definition for the Pokémon Red agent."""

from __future__ import annotations

from enum import IntEnum


class Action(IntEnum):
    """All inputs the agent can send to the Game Boy."""

    NOOP   = 0
    UP     = 1
    DOWN   = 2
    LEFT   = 3
    RIGHT  = 4
    A      = 5
    B      = 6
    START  = 7
    SELECT = 8


# Convenience list – useful for sampling and iteration.
ALL_ACTIONS: list[Action] = list(Action)

# Movement-only subset, useful for the heuristic agent in overworld mode.
MOVEMENT_ACTIONS: list[Action] = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]

# Battle-relevant actions.
BATTLE_ACTIONS: list[Action] = [Action.UP, Action.DOWN, Action.A, Action.B]

N_ACTIONS: int = len(ALL_ACTIONS)
