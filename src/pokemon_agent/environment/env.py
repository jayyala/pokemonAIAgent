"""Gymnasium-compatible environment wrapper for Pokémon Red.

Wraps the emulator + state parser + reward function into a standard
``gymnasium.Env`` so that any off-the-shelf RL library can train on it.

Observation space:  Box(10,) – float32 feature vector from GameState.as_vector()
Action space:       Discrete(9)
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces

    _GYM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _GYM_AVAILABLE = False

from ..emulator.base import Emulator
from .actions import N_ACTIONS
from .rewards import RewardConfig, compute_reward
from .state import GameState, read_state


class PokemonRedEnv:
    """Gym-compatible wrapper around the Pokémon Red emulator.

    Usage::

        env = PokemonRedEnv(emulator)
        obs, info = env.reset()
        for _ in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
    """

    metadata = {"render_modes": ["rgb_array"]}

    OBS_DIM = 10   # length of GameState.as_vector()

    def __init__(
        self,
        emulator: Emulator,
        reward_config: Optional[RewardConfig] = None,
        max_steps: int = 20_480,
        render_mode: Optional[str] = None,
    ) -> None:
        self._emulator = emulator
        self._reward_cfg = reward_config or RewardConfig()
        self._max_steps = max_steps
        self.render_mode = render_mode

        self._prev_state: Optional[GameState] = None
        self._curr_state: Optional[GameState] = None
        self._episode_steps: int = 0
        self._episode_reward: float = 0.0

        if _GYM_AVAILABLE:
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(self.OBS_DIM,), dtype=np.float32
            )
            self.action_space = spaces.Discrete(N_ACTIONS)

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self._emulator.reset()
        self._curr_state = read_state(self._emulator)
        self._prev_state = self._curr_state
        self._episode_steps = 0
        self._episode_reward = 0.0
        obs = self._make_obs()
        return obs, {}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._emulator.step(action)
        self._prev_state = self._curr_state
        self._curr_state = read_state(self._emulator)
        self._episode_steps += 1

        assert self._prev_state is not None
        reward = compute_reward(self._prev_state, self._curr_state, self._reward_cfg)
        self._episode_reward += reward

        terminated = self._curr_state.is_fainted
        truncated  = self._episode_steps >= self._max_steps

        info: dict[str, Any] = {
            "episode_steps":  self._episode_steps,
            "episode_reward": self._episode_reward,
            "badge_count":    self._curr_state.badge_count,
            "pokedex_seen":   self._curr_state.pokedex_seen,
            "map_id":         self._curr_state.map_id,
        }
        return self._make_obs(), reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "rgb_array":
            return self._emulator.screen_rgb()
        return None

    def close(self) -> None:
        self._emulator.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_obs(self) -> np.ndarray:
        assert self._curr_state is not None
        return np.array(self._curr_state.as_vector(), dtype=np.float32)
