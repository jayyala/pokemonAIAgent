"""Tests for the experience replay buffer."""

import numpy as np
import pytest

from pokemon_agent.training.replay_buffer import ReplayBuffer


def _dummy_transition(obs_dim: int = 10):
    s  = np.zeros(obs_dim, dtype=np.float32)
    ns = np.ones(obs_dim, dtype=np.float32)
    return s, 0, 1.0, ns, False


def test_push_and_len():
    buf = ReplayBuffer(capacity=100)
    assert len(buf) == 0
    buf.push(*_dummy_transition())
    assert len(buf) == 1


def test_capacity_eviction():
    buf = ReplayBuffer(capacity=5)
    for _ in range(10):
        buf.push(*_dummy_transition())
    assert len(buf) == 5


def test_sample_shape():
    buf = ReplayBuffer(capacity=100, seed=0)
    for _ in range(20):
        buf.push(*_dummy_transition())
    states, actions, rewards, next_states, dones = buf.sample(8)
    assert states.shape      == (8, 10)
    assert actions.shape     == (8,)
    assert rewards.shape     == (8,)
    assert next_states.shape == (8, 10)
    assert dones.shape       == (8,)


def test_sample_dtypes():
    buf = ReplayBuffer(capacity=100, seed=1)
    for _ in range(20):
        buf.push(*_dummy_transition())
    states, actions, rewards, next_states, dones = buf.sample(4)
    assert states.dtype      == np.float32
    assert actions.dtype     == np.int64
    assert rewards.dtype     == np.float32
    assert next_states.dtype == np.float32
    assert dones.dtype       == np.float32


def test_sample_raises_when_too_small():
    buf = ReplayBuffer(capacity=100)
    buf.push(*_dummy_transition())
    with pytest.raises(ValueError, match="Cannot sample"):
        buf.sample(10)


def test_repr():
    buf = ReplayBuffer(capacity=50)
    assert "ReplayBuffer" in repr(buf)
