"""Tests for agent implementations."""

import numpy as np
import pytest

from pokemon_agent.agents.random_agent import RandomAgent
from pokemon_agent.agents.heuristic_agent import HeuristicAgent
from pokemon_agent.environment.actions import N_ACTIONS


OBS_DIM = 10


def _obs(in_battle: bool = False) -> np.ndarray:
    o = np.zeros(OBS_DIM, dtype=np.float32)
    o[3] = 0.5 if in_battle else 0.0  # battle_type field
    return o


# ---------------------------------------------------------------------------
# RandomAgent
# ---------------------------------------------------------------------------

class TestRandomAgent:
    def test_action_in_range(self):
        agent = RandomAgent(seed=0)
        for _ in range(100):
            a = agent.select_action(_obs())
            assert 0 <= a < N_ACTIONS

    def test_reproducible_with_seed(self):
        a1 = [RandomAgent(seed=7).select_action(_obs()) for _ in range(20)]
        a2 = [RandomAgent(seed=7).select_action(_obs()) for _ in range(20)]
        assert a1 == a2

    def test_different_seeds_differ(self):
        a1 = [RandomAgent(seed=0).select_action(_obs()) for _ in range(20)]
        a2 = [RandomAgent(seed=1).select_action(_obs()) for _ in range(20)]
        assert a1 != a2


# ---------------------------------------------------------------------------
# HeuristicAgent
# ---------------------------------------------------------------------------

class TestHeuristicAgent:
    def test_action_in_range(self):
        agent = HeuristicAgent(seed=0)
        for _ in range(100):
            a = agent.select_action(_obs())
            assert 0 <= a < N_ACTIONS

    def test_battle_action_in_range(self):
        agent = HeuristicAgent(seed=0)
        for _ in range(100):
            a = agent.select_action(_obs(in_battle=True))
            assert 0 <= a < N_ACTIONS

    def test_update_returns_empty(self):
        agent = HeuristicAgent()
        result = agent.update()
        assert result == {}

    def test_high_battle_a_prob(self):
        """Agent should press A frequently in battle."""
        from pokemon_agent.environment.actions import Action

        agent   = HeuristicAgent(battle_a_prob=1.0, seed=0)
        actions = [agent.select_action(_obs(in_battle=True)) for _ in range(50)]
        assert all(a == Action.A for a in actions)
