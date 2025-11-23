"""Experience replay buffer for off-policy RL algorithms.

Stores (state, action, reward, next_state, done) transitions in a fixed-
capacity ring buffer and exposes a vectorised ``sample()`` method that
returns numpy arrays suitable for batched gradient updates.
"""

from __future__ import annotations

from collections import deque
from typing import NamedTuple

import numpy as np


class Transition(NamedTuple):
    """A single environment transition."""

    state:      np.ndarray   # float32 (obs_dim,)
    action:     int
    reward:     float
    next_state: np.ndarray   # float32 (obs_dim,)
    done:       bool


class ReplayBuffer:
    """Uniform random experience replay buffer.

    Args:
        capacity: Maximum number of transitions to store.  When the buffer
                  is full the oldest transition is evicted.
        seed:     Optional RNG seed for reproducible sampling.
    """

    def __init__(self, capacity: int = 100_000, seed: int | None = None) -> None:
        self._buffer: deque[Transition] = deque(maxlen=capacity)
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        """Add a transition to the buffer."""
        self._buffer.append(
            Transition(
                state      = np.asarray(state,      dtype=np.float32),
                action     = int(action),
                reward     = float(reward),
                next_state = np.asarray(next_state, dtype=np.float32),
                done       = bool(done),
            )
        )

    def sample(self, batch_size: int) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """Return a random batch as stacked numpy arrays.

        Args:
            batch_size: Number of transitions to sample (with replacement).

        Returns:
            Tuple of (states, actions, rewards, next_states, dones), each
            with shape (batch_size, *field_shape) and appropriate dtype.

        Raises:
            ValueError: If the buffer contains fewer entries than
                        ``batch_size``.
        """
        if len(self) < batch_size:
            raise ValueError(
                f"Cannot sample {batch_size} transitions from a buffer "
                f"of size {len(self)}."
            )

        indices = self._rng.integers(0, len(self._buffer), size=batch_size)
        batch   = [self._buffer[i] for i in indices]

        states      = np.stack([t.state      for t in batch])           # (B, obs_dim)
        actions     = np.array([t.action     for t in batch], np.int64) # (B,)
        rewards     = np.array([t.reward     for t in batch], np.float32)
        next_states = np.stack([t.next_state for t in batch])
        dones       = np.array([t.done       for t in batch], np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return (
            f"ReplayBuffer(size={len(self)}, "
            f"capacity={self._buffer.maxlen})"
        )
