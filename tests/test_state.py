"""Tests for game-state parsing."""

import pytest

from pokemon_agent.emulator.mock_emulator import MockEmulator, MockEmulatorConfig
from pokemon_agent.environment.state import GameState, read_state


def _make_emulator(**kwargs) -> MockEmulator:
    cfg = MockEmulatorConfig(**kwargs)
    return MockEmulator(cfg)


def test_read_state_returns_game_state():
    emu   = _make_emulator()
    state = read_state(emu)
    assert isinstance(state, GameState)


def test_hp_fraction_clipped():
    emu   = _make_emulator(initial_hp=50, max_hp=100)
    state = read_state(emu)
    assert 0.0 <= state.hp_fraction <= 1.0


def test_fainted_flag():
    emu = MockEmulator()
    emu.set_hp(0)
    state = read_state(emu)
    assert state.is_fainted


def test_not_fainted_when_healthy():
    emu   = _make_emulator(initial_hp=100, max_hp=100)
    state = read_state(emu)
    assert not state.is_fainted


def test_as_vector_length():
    emu   = _make_emulator()
    state = read_state(emu)
    vec   = state.as_vector()
    assert len(vec) == 10


def test_as_vector_values_in_range():
    emu   = _make_emulator(initial_hp=50, max_hp=100)
    state = read_state(emu)
    for v in state.as_vector():
        assert -0.01 <= v <= 1.01, f"Value {v} out of expected range"


def test_badge_count():
    # 3 badges = bitmask 0b00000111 = 7
    emu = MockEmulator(MockEmulatorConfig(memory_map={0xD356: 0b00000111}))
    state = read_state(emu)
    assert state.badge_count == 3


def test_in_battle_true():
    emu   = MockEmulator(MockEmulatorConfig(memory_map={0xD057: 1}))
    state = read_state(emu)
    assert state.in_battle


def test_in_battle_false():
    emu   = MockEmulator(MockEmulatorConfig(memory_map={0xD057: 0}))
    state = read_state(emu)
    assert not state.in_battle
