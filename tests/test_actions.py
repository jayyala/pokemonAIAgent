"""Tests for the action space."""

import pytest

from pokemon_agent.environment.actions import (
    Action,
    ALL_ACTIONS,
    BATTLE_ACTIONS,
    MOVEMENT_ACTIONS,
    N_ACTIONS,
)


def test_action_count():
    assert len(ALL_ACTIONS) == 9
    assert N_ACTIONS == 9


def test_action_members():
    assert Action.NOOP   in ALL_ACTIONS
    assert Action.UP     in ALL_ACTIONS
    assert Action.DOWN   in ALL_ACTIONS
    assert Action.LEFT   in ALL_ACTIONS
    assert Action.RIGHT  in ALL_ACTIONS
    assert Action.A      in ALL_ACTIONS
    assert Action.B      in ALL_ACTIONS
    assert Action.START  in ALL_ACTIONS
    assert Action.SELECT in ALL_ACTIONS


def test_action_values():
    assert Action.NOOP   == 0
    assert Action.UP     == 1
    assert Action.DOWN   == 2
    assert Action.LEFT   == 3
    assert Action.RIGHT  == 4
    assert Action.A      == 5
    assert Action.B      == 6
    assert Action.START  == 7
    assert Action.SELECT == 8


def test_movement_actions_subset():
    for a in MOVEMENT_ACTIONS:
        assert a in ALL_ACTIONS
    assert Action.NOOP not in MOVEMENT_ACTIONS
    assert Action.A    not in MOVEMENT_ACTIONS


def test_battle_actions_subset():
    for a in BATTLE_ACTIONS:
        assert a in ALL_ACTIONS
