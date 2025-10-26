"""Reward shaping for the Pokémon Red agent.

The reward signal is decomposed into several sparse and dense components
so that the agent receives useful gradients even in early training when
meaningful game events are rare.

    R(t) = r_hp + r_explore + r_badge + r_pokedex + r_level + r_faint

Each component is scaled independently via ``RewardConfig`` so the
overall signal can be tuned without touching agent or training code.
"""

from __future__ import annotations

from dataclasses import dataclass

from .state import GameState


@dataclass
class RewardConfig:
    """Coefficients for each reward component."""

    # Dense survival signal: fraction of max HP remaining each step.
    hp_scale: float = 0.01

    # Sparse event: agent earns this multiplied by number of new map tiles
    # visited since last step (encourages exploration).
    explore_scale: float = 0.02

    # Sparse event: badge obtained (+N per new badge).
    badge_scale: float = 5.0

    # Sparse event: new Pokémon added to Pokédex.
    pokedex_seen_scale: float = 0.5
    pokedex_owned_scale: float = 1.0

    # Sparse event: lead Pokémon gains a level.
    level_scale: float = 2.0

    # Terminal penalty when lead Pokémon faints.
    faint_penalty: float = -5.0


def compute_reward(
    prev: GameState,
    curr: GameState,
    config: RewardConfig | None = None,
) -> float:
    """Compute the shaped reward between two consecutive game states.

    Args:
        prev:   State at time *t-1*.
        curr:   State at time *t*.
        config: Reward coefficient configuration.  Defaults to
                ``RewardConfig()`` if not provided.

    Returns:
        Scalar float reward.
    """
    cfg = config or RewardConfig()
    reward: float = 0.0

    # --- HP survival signal (dense) -----------------------------------
    reward += cfg.hp_scale * curr.hp_fraction

    # --- Exploration --------------------------------------------------
    map_changed = int(curr.map_id != prev.map_id)
    reward += cfg.explore_scale * map_changed

    # --- Badges (sparse, per new badge) --------------------------------
    new_badges = curr.badge_count - prev.badge_count
    if new_badges > 0:
        reward += cfg.badge_scale * new_badges

    # --- Pokédex (sparse) ---------------------------------------------
    new_seen = curr.pokedex_seen - prev.pokedex_seen
    if new_seen > 0:
        reward += cfg.pokedex_seen_scale * new_seen

    new_owned = curr.pokedex_owned - prev.pokedex_owned
    if new_owned > 0:
        reward += cfg.pokedex_owned_scale * new_owned

    # --- Level-up (sparse) --------------------------------------------
    new_levels = curr.level - prev.level
    if new_levels > 0:
        reward += cfg.level_scale * new_levels

    # --- Faint penalty ------------------------------------------------
    if curr.is_fainted and not prev.is_fainted:
        reward += cfg.faint_penalty

    return reward
